from pathlib import Path


def get_config():
    return {
        "train_batch_size": 32,
        "val_batch_size": 32,
        "test_batch_size": 64,
        "num_epochs": 5,
        "lr": 1e-4,
        "dropout": 0.1,
        "weight_decay": 0.01,
        "seq_len": 256,
        "d_model": 256,
        "N": 4,
        "h": 8,
        "d_ff": 512,
        "architecture": "transformer-base",
        "model_basename": "combinators-llm-test",
        "preload": None,
        "term_tokenizer_file": "term-tokenizer.json",
        "type_tokenizer_file": "type-tokenizer.json",
        "val_accuracy_early_stop": 0.1,
        "lean_batch_size": 32,
    }


def get_weights_file_path(config, epoch):
    model_folder = "weights"
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)
