import torch
import random
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from .utils import causal_mask


class CombinatorsDataset(Dataset):

    def __init__(self, split: str, seq_len: int) -> None:
        super().__init__()

        self.type_tokenizer = Tokenizer.from_file(
            "combinators_llm/tokenizers/type-tokenizer.json"
        )
        self.term_tokenizer = Tokenizer.from_file(
            "combinators_llm/tokenizers/term-tokenizer.json"
        )
        self.seq_len = seq_len

        dataset = load_dataset("carlosFerLo/combinators-dataset", split=split)

        dataset = dataset.filter(
            lambda x: len(self.type_tokenizer.encode(x["type"]).ids) < seq_len - 2
        )
        dataset = dataset.filter(
            lambda x: len(self.term_tokenizer.encode(x["term"]).ids) < seq_len - 1
        )

        if split in ["test", "validation"]:

            seen = set()

            def filter_unique(example):
                if example["type"] in seen:
                    return False
                seen.add(example["type"])
                return True

            dataset = dataset.filter(filter_unique)

        self.ds = dataset.shuffle()

        self.enc_sos_token = torch.tensor(
            [self.type_tokenizer.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.enc_eos_token = torch.tensor(
            [self.type_tokenizer.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.enc_pad_token = torch.tensor(
            [self.type_tokenizer.token_to_id("[PAD]")], dtype=torch.int64
        )

        self.dec_sos_token = torch.tensor(
            [self.term_tokenizer.token_to_id("[SOS]")], dtype=torch.int64
        )
        self.dec_eos_token = torch.tensor(
            [self.term_tokenizer.token_to_id("[EOS]")], dtype=torch.int64
        )
        self.dec_pad_token = torch.tensor(
            [self.term_tokenizer.token_to_id("[PAD]")], dtype=torch.int64
        )

    def __len__(self):
        return len(self.ds)  # type: ignore

    def __getitem__(self, index):
        type_term_pair = self.ds[index]
        type_text = type_term_pair["type"]
        term_text = type_term_pair["term"]

        enc_input_tokens = self.type_tokenizer.encode(type_text).ids
        dec_input_tokens = self.term_tokenizer.encode(term_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        encoder_input = torch.cat(
            [
                self.enc_sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.enc_eos_token,
                torch.tensor(
                    [self.enc_pad_token] * enc_num_padding_tokens, dtype=torch.int64
                ),
            ]
        )

        decoder_input = torch.cat(
            [
                self.dec_sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor(
                    [self.dec_pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ]
        )

        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.dec_eos_token,
                torch.tensor(
                    [self.dec_pad_token] * dec_num_padding_tokens, dtype=torch.int64
                ),
            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            "encoder_mask": (encoder_input != self.enc_pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int(),  # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.dec_pad_token)
            .unsqueeze(0)
            .unsqueeze(0)
            .int()
            & causal_mask(
                decoder_input.size(0),
            ),  # (1, 1, seq_len) & (1, seq_len, seq_len)
            "label": label,
            "type_text": type_text,
            "term_text": term_text,
        }


def get_ds(config):
    train = CombinatorsDataset("train", config["seq_len"])
    val = CombinatorsDataset("validation", config["seq_len"])
    test = CombinatorsDataset("test", config["seq_len"])

    train_dataloader = DataLoader(
        train, batch_size=config["train_batch_size"], shuffle=True
    )
    val_dataloader = DataLoader(val, batch_size=config["val_batch_size"], shuffle=True)
    test_dataloader = DataLoader(
        test, batch_size=config["test_batch_size"], shuffle=True
    )

    return train_dataloader, val_dataloader, test_dataloader
