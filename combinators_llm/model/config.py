from pathlib import Path

def get_config () :
    return {
        "train_batch_size": 16,
        "val_batch_size": 1,
        "test_batch_size": 1,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 75,
        "d_model": 64,
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "term_tokenizer_file": "term-tokenizer.json",
        "type_tokenizer_file": "type-tokenizer.json"
    }
    
def get_weights_file_path (config, epoch: str) :
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)