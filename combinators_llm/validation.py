import torch
import logging
from tqdm import tqdm
from typing import List, Tuple
from pathlib import Path
from datetime import datetime
import json

from .generators import greedy_decode_batch, beam_search_decode_batch
from .utils.lean import check_proof_batch
from .build import get_model
from .dataset import get_ds
from .tokenizers import get_tokenizer
from .config import get_config


def run_validation(
    val_name,
    model,
    validation_ds,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
    global_step=None,
    run=None,
) -> float:
    """Run validation using greedy decoding."""
    model.eval()
    logger = logging.getLogger(__name__)

    count = 0
    total = 0

    with torch.no_grad():
        batch_iterator = tqdm(validation_ds, desc=f"Running {val_name} (greedy)")
        print_batch = True
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            model_out = greedy_decode_batch(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_tgt,
                max_len,
                device,
            )
            model_out = model_out.detach().cpu().numpy()

            pairs: List[Tuple[str, str]] = []

            for i in range(encoder_input.size(0)):
                type_text: str = batch["type_text"][i]

                term_text = tokenizer_tgt.decode(model_out[i])
                term_text: str = term_text.replace("[SOS]", "").split("[EOS]")[0]

                pairs.append((type_text, term_text))

            if print_batch:
                for type, term in pairs:
                    logger.debug(f"TYPE: {type} | TERM: {term}")
                print_batch = False

            total += len(batch["type_text"])
            res = check_proof_batch(pairs)
            count += sum(res)

            batch_iterator.set_postfix({"acc": f"{count / total :6.3f}"})

    results = count / total

    if run is not None and global_step is not None:
        run.log({f"{val_name}_greedy_acc": results}, global_step)

    logger.info(f"Greedy validation accuracy for {val_name}: {results:.4f}")

    return results


def run_validation_beam_search(
    val_name,
    model,
    validation_ds,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
    config,
    global_step=None,
    run=None,
) -> float:
    """Run validation using beam search with batched Lean checking.

    For each input, generates multiple candidate beams. An input is considered
    correct if at least one of its beams passes Lean validation.

    Args:
        val_name: Name of the validation set
        model: The transformer model
        validation_ds: Validation dataloader
        tokenizer_src: Source tokenizer
        tokenizer_tgt: Target tokenizer
        max_len: Maximum sequence length
        device: Device to run on
        config: Configuration dictionary containing beam search parameters
        global_step: Current training step (for wandb logging)
        run: Wandb run object (for logging)

    Returns:
        Accuracy as a float (correct inputs / total inputs)
    """
    model.eval()
    logger = logging.getLogger(__name__)

    count = 0
    total = 0

    # Extract beam search parameters from config
    beam_size = config.get("beam_size", 4)
    length_penalty = config.get("length_penalty", 1.0)
    max_num_sequences = config.get("max_num_sequences", 100)
    batch_beam_size = config.get("batch_beam_size", 8)

    with torch.no_grad():
        batch_iterator = tqdm(validation_ds, desc=f"Running {val_name} (beam search)")
        print_batch = True

        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

            # beam_search_decode_batch returns List[List[Tuple[tensor, score]]]
            # Outer list: one entry per input in batch
            # Inner list: beams for that input
            batch_beams = beam_search_decode_batch(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_tgt,
                max_len,
                device,
                beam_size=beam_size,
                length_penalty=length_penalty,
                max_num_sequences=max_num_sequences,
                batch_beam_size=batch_beam_size,
            )

            # Process each input and its beams
            for i, beams in enumerate(batch_beams):
                type_text: str = batch["type_text"][i]
                pairs: List[Tuple[str, str]] = []

                # Decode all beams for this input
                for beam_seq, score in beams:
                    term_text = tokenizer_tgt.decode(beam_seq.detach().cpu().numpy())
                    term_text = term_text.replace("[SOS]", "").split("[EOS]")[0]
                    pairs.append((type_text, term_text))

                if print_batch and i == 0:
                    # Print first input's beams for debugging
                    logger.debug(f"TYPE: {type_text}")
                    for j, (beam_seq, score) in enumerate(beams[:5]):  # Show top 5
                        term_text = tokenizer_tgt.decode(
                            beam_seq.detach().cpu().numpy()
                        )
                        term_text = term_text.replace("[SOS]", "").split("[EOS]")[0]
                        logger.debug(f"  Beam {j+1} (score={score:.4f}): {term_text}")
                    print_batch = False

                # Validate terms of this input
                res = check_proof_batch(pairs)

                count += any(res)
                total += 1

            batch_iterator.set_postfix(
                {"acc": f"{count / total if total > 0 else 0 :6.3f}"}
            )

    results = count / total if total > 0 else 0.0

    if run is not None and global_step is not None:
        run.log({f"{val_name}_beam_acc": results}, global_step)

    logger.info(f"Beam search validation accuracy for {val_name}: {results:.4f}")

    return results


def save_validation_results(
    results: dict, output_file: str = "validation_results.json"
):
    """Save validation results to a JSON file with individual timestamps per partition.

    Only updates the partitions that were tested, preserving existing data for others.
    Supports both greedy and beam search accuracy metrics.
    """

    logger = logging.getLogger(__name__)
    output_path = Path(output_file)
    current_time = datetime.now().isoformat()

    # Load existing results or create new structure
    existing_data: dict = {}
    if output_path.exists():
        with open(output_path, "r") as f:
            try:
                existing_data = json.load(f)
                # If old format (list), convert to new format
                if isinstance(existing_data, list):
                    existing_data = {}
            except json.JSONDecodeError:
                existing_data = {}

    # Ensure all partitions exist in the structure
    for partition in ["train", "validation", "test"]:
        if partition not in existing_data:
            existing_data[partition] = None

    # Update only the tested partitions with new results and timestamps
    for partition, result in results.items():
        existing_data[partition] = {
            "greedy_accuracy": result.get("greedy_accuracy"),
            "beam_accuracy": result.get("beam_accuracy"),
            "total_batches": result["total_batches"],
            "last_validated": current_time,
        }

    # Save to file
    with open(output_path, "w") as f:
        json.dump(existing_data, f, indent=2)

    logger.info(f"Results saved to {output_file}")

    # Print what was updated
    for partition in results.keys():
        logger.info(
            f"Updated {partition} validation results (timestamp: {current_time})"
        )


def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logging.info("Loading model config...")
    config = get_config()
    logging.info(f"Model Config:\n{config}")

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Ask user which partitions to validate
    print("\n" + "=" * 50)
    print("VALIDATION SCRIPT")
    print("=" * 50)
    print("\nSelect partition(s) to validate:")
    print("1. Train")
    print("2. Validation")
    print("3. Test")
    print("4. All")
    print("=" * 50)

    partition_choice = input("\nEnter your choice (1-4): ").strip()

    partitions_to_run = []
    if partition_choice == "1":
        partitions_to_run = ["train"]
    elif partition_choice == "2":
        partitions_to_run = ["validation"]
    elif partition_choice == "3":
        partitions_to_run = ["test"]
    elif partition_choice == "4":
        partitions_to_run = ["train", "validation", "test"]
    else:
        logging.error("Invalid choice. Exiting.")
        return

    # Ask user which validation method(s) to use
    print("\n" + "=" * 50)
    print("Select validation method(s):")
    print("1. Greedy")
    print("2. Beam Search")
    print("3. Both")
    print("=" * 50)

    method_choice = input("\nEnter your choice (1-3): ").strip()

    run_greedy = False
    run_beam = False
    if method_choice == "1":
        run_greedy = True
    elif method_choice == "2":
        run_beam = True
    elif method_choice == "3":
        run_greedy = True
        run_beam = True
    else:
        logging.error("Invalid choice. Exiting.")
        return

    logging.info(f"Running validation on: {', '.join(partitions_to_run)}")
    methods = []
    if run_greedy:
        methods.append("greedy")
    if run_beam:
        methods.append("beam search")
    logging.info(f"Using methods: {', '.join(methods)}")

    # Load data
    logging.info("Loading datasets...")
    train_dataloader, val_dataloader, test_dataloader = get_ds(config)

    # Load tokenizers
    logging.info("Loading tokenizers...")
    src_tokenizer = get_tokenizer("type")
    tgt_tokenizer = get_tokenizer("term")

    # Build model
    logging.info("Building model...")
    model = get_model(
        config,
        src_tokenizer.get_vocab_size(),
        tgt_tokenizer.get_vocab_size(),
        src_tokenizer.token_to_id("[PAD]"),
        tgt_tokenizer.token_to_id("[PAD]"),
    ).to(device)

    # Load model weights
    model_path = Path("./combinators_llm/combinators-llm.bin")
    if not model_path.exists():
        logging.error(f"Model weights not found at {model_path}")
        return

    logging.info(f"Loading model weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Run validation on selected partitions
    results = {}

    dataloader_map = {
        "train": train_dataloader,
        "validation": val_dataloader,
        "test": test_dataloader,
    }

    for partition in partitions_to_run:
        logging.info(f"\n{'='*50}")
        logging.info(f"Running validation on {partition} set...")
        logging.info(f"{'='*50}")

        dataloader = dataloader_map[partition]

        partition_results = {
            "total_batches": len(dataloader),
        }

        # Run greedy validation if selected
        if run_greedy:
            logging.info(f"Running greedy validation for {partition}...")
            greedy_accuracy = run_validation(
                partition,
                model,
                dataloader,
                src_tokenizer,
                tgt_tokenizer,
                config["seq_len"],
                device,
            )
            partition_results["greedy_accuracy"] = greedy_accuracy  # type: ignore

        # Run beam search validation if selected
        if run_beam:
            logging.info(f"Running beam search validation for {partition}...")
            beam_accuracy = run_validation_beam_search(
                partition,
                model,
                dataloader,
                src_tokenizer,
                tgt_tokenizer,
                config["seq_len"],
                device,
                config,
            )
            partition_results["beam_accuracy"] = beam_accuracy  # type: ignore

        results[partition] = partition_results

    # Save results
    save_validation_results(results)

    # Load the complete results file to show all partitions
    output_path = Path("validation_results.json")
    if output_path.exists():
        with open(output_path, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    # Print summary with timestamps
    print("\n" + "=" * 85)
    print("VALIDATION RESULTS SUMMARY")
    print("=" * 85)
    print(
        f"{'Partition':<15} {'Greedy Acc':<15} {'Beam Acc':<15} {'Last Validated':<30}"
    )
    print("-" * 85)

    for partition in ["train", "validation", "test"]:
        if partition in all_results and all_results[partition] is not None:
            greedy_acc = all_results[partition].get("greedy_accuracy")
            beam_acc = all_results[partition].get("beam_accuracy")
            timestamp = all_results[partition]["last_validated"]

            # Format timestamp for better readability
            try:
                dt = datetime.fromisoformat(timestamp)
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                formatted_time = timestamp

            # Format accuracies
            greedy_str = f"{greedy_acc:.4f}" if greedy_acc is not None else "N/A"
            beam_str = f"{beam_acc:.4f}" if beam_acc is not None else "N/A"

            # Mark newly tested partitions
            marker = " *" if partition in results else ""
            print(
                f"{partition.capitalize():<15} {greedy_str:<15} {beam_str:<15} {formatted_time:<30}{marker}"
            )
        else:
            print(
                f"{partition.capitalize():<15} {'N/A':<15} {'N/A':<15} {'Not tested yet':<30}"
            )

    print("-" * 85)
    if results:
        print("* = Updated in this run")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
