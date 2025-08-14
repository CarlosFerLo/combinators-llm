from pathlib import Path

def get_config () :
    return {
        "train_batch_size": 16,
        "val_batch_size": 1,
        "test_batch_size": 1,
        "num_epochs": 20,
        "lr": 10**-4,
        "dropout": 0.1,
        "seq_len": 75,
        "d_model": 64,
        "N": 6,
        "h": 4, 
        "d_ff": 128,
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "term_tokenizer_file": "term-tokenizer.json",
        "type_tokenizer_file": "type-tokenizer.json"
    }