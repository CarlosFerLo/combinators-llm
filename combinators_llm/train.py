import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from .build import build_transformer
from .dataset import get_ds
import logging
from pathlib import Path
from .tokenizers import get_tokenizer
from .config import get_config, get_weights_file_path
from .validation import run_validation
from dotenv import load_dotenv
import os

load_dotenv()

print("Hello from train.py")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(threadName)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    filename="train.log",
    filemode="a",
)


def get_model(config, vocab_src_len, vocab_tgt_len, src_pad_idx, tgt_pad_idx):
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"],
        config["seq_len"],
        src_pad_idx,
        tgt_pad_idx,
        config["d_model"],
        config["N"],
        config["h"],
        config["d_ff"],
        config["dropout"],
    )
    return model


def train_model(config):

    # Initialize WandB
    run = wandb.init(
        entity=os.environ["WANDB_ENTITY"],
        project=os.environ["WANDB_PROJECT"],
        config=config,
    )

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    Path("./weights").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, test_dataloader = get_ds(config)
    src_tokenizer = get_tokenizer("type")
    tgt_tokenizer = get_tokenizer("term")

    model = get_model(
        config,
        src_tokenizer.get_vocab_size(),
        tgt_tokenizer.get_vocab_size(),
        src_tokenizer.token_to_id("[PAD]"),
        tgt_tokenizer.token_to_id("[PAD]"),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        logging.info(f"Preloading model {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]
    else:
        logging.info("Initializing weights")
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=src_tokenizer.token_to_id("[PAD]"), label_smoothing=0.1
    ).to(device)

    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)  # (batch, seq_len)
            decoder_input = batch["decoder_input"].to(device)  # (batch, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)  # (batch, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(
                device
            )  # (batch, 1, seq_len, seq_len)

            encoder_output = model.encode(
                encoder_input, encoder_mask
            )  # (batch, seq_len, d_model)
            decoder_output = model.decode(
                encoder_output, encoder_mask, decoder_input, decoder_mask
            )  # (batch, seq_len, d_model)
            projection_output = model.project(
                decoder_output
            )  # (batch, seq_len, tgt_vocab_size)

            label = batch["label"].to(device)  # (batch, seq_len)

            # (batch, seq_len, tgt_vocab_size) --> (batch * seq_len, tgt_vocab_size)
            loss = loss_fn(
                projection_output.view(-1, tgt_tokenizer.get_vocab_size()),
                label.view(-1),
            )

            run.log({"train_loss": loss.item()}, global_step)

            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            logging.info(f"global_step: {global_step}, loss: {loss.item():6.3f}")

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # Save the model after each epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )

        logging.info("Running validation...")
        run_validation(
            "validation",
            model,
            val_dataloader,
            src_tokenizer,
            tgt_tokenizer,
            config["seq_len"],
            device,
            global_step,
            run,
        )

    logging.info("Running testing...")
    run_validation(
        "test",
        model,
        test_dataloader,
        src_tokenizer,
        tgt_tokenizer,
        config["seq_len"],
        device,
        global_step,
        run,
    )

    run.finish()


def main():
    logging.info("Loading model config...")
    config = get_config()
    logging.info(f"Model Config:\n{config}")
    train_model(config)


if __name__ == "__main__":
    main()
