from pathlib import Path


def get_config():
    return {
        "train_batch_size": 16,
        "val_batch_size": 8,
        "test_batch_size": 8,
        "num_epochs": 1,
        "lr": 1e-5,
        "dropout": 0.1,
        "seq_len": 256,
        "d_model": 80,
        "N": 6,
        "h": 8,
        "d_ff": 160,
        "architecture": "transformer-base",
        "model_basename": "combinators-llm-test",
        "preload": None,
        "term_tokenizer_file": "term-tokenizer.json",
        "type_tokenizer_file": "type-tokenizer.json",
    }


def get_weights_file_path(config, epoch):
    model_folder = "weights"
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)
