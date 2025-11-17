import torch
import logging
from tqdm import tqdm
from typing import List, Tuple
from pathlib import Path
from datetime import datetime
import json

from .generators import greedy_decode_batch
from .utils.lean import check_proof_batch
from .build import get_model
from .dataset import get_ds
from .tokenizers import get_tokenizer
from .config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(threadName)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


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
    model.eval()
    logging.getLogger(__name__)

    count = 0
    total = 0

    with torch.no_grad():
        batch_iterator = tqdm(validation_ds, desc=f"Running {val_name}")
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
                    logging.debug(f"TYPE: {type} | TERM: {term}")
                print_batch = False

            total += len(batch["type_text"])
            res = check_proof_batch(pairs)
            count += sum(res)

            batch_iterator.set_postfix({"acc": f"{count / total :6.3f}"})

    results = count / total

    if run is not None and global_step is not None:
        run.log({f"{val_name}_acc": results}, global_step)

    logging.info(f"Validation accuracy for {val_name}: {results:.4f}")

    return results


def save_validation_results(
    results: dict, output_file: str = "validation_results.json"
):
    """Save validation results to a JSON file with individual timestamps per partition.

    Only updates the partitions that were tested, preserving existing data for others.
    """
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
            "accuracy": result["accuracy"],
            "total_batches": result["total_batches"],
            "last_validated": current_time,
        }

    # Save to file
    with open(output_path, "w") as f:
        json.dump(existing_data, f, indent=2)

    logging.info(f"Results saved to {output_file}")

    # Print what was updated
    for partition in results.keys():
        logging.info(
            f"Updated {partition} validation results (timestamp: {current_time})"
        )


def main():
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

    choice = input("\nEnter your choice (1-4): ").strip()

    partitions_to_run = []
    if choice == "1":
        partitions_to_run = ["train"]
    elif choice == "2":
        partitions_to_run = ["validation"]
    elif choice == "3":
        partitions_to_run = ["test"]
    elif choice == "4":
        partitions_to_run = ["train", "validation", "test"]
    else:
        logging.error("Invalid choice. Exiting.")
        return

    logging.info(f"Running validation on: {', '.join(partitions_to_run)}")

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
        accuracy = run_validation(
            partition,
            model,
            dataloader,
            src_tokenizer,
            tgt_tokenizer,
            config["seq_len"],
            device,
        )

        results[partition] = {
            "accuracy": accuracy,
            "total_batches": len(dataloader),
        }

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
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Partition':<15} {'Accuracy':<12} {'Last Validated':<30}")
    print("-" * 70)

    for partition in ["train", "validation", "test"]:
        if partition in all_results and all_results[partition] is not None:
            acc = all_results[partition]["accuracy"]
            timestamp = all_results[partition]["last_validated"]
            # Format timestamp for better readability
            try:
                dt = datetime.fromisoformat(timestamp)
                formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                formatted_time = timestamp

            # Mark newly tested partitions
            marker = " *" if partition in results else ""
            print(
                f"{partition.capitalize():<15} {acc:<12.4f} {formatted_time:<30}{marker}"
            )
        else:
            print(f"{partition.capitalize():<15} {'N/A':<12} {'Not tested yet':<30}")

    print("-" * 70)
    if results:
        print("* = Updated in this run")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
