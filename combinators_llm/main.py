import torch
from typing import Dict, Any, List, Union
from tokenizers import Tokenizer

from .modules import Transformer
from .build import get_model
from .config import get_config
from .tokenizers import get_tokenizer
from .generators.greedy import greedy_decode, greedy_decode_batch


class CombinatorsLlm:

    config: Dict[str, Any]
    transformer: Transformer
    device: torch.device

    term_tokenizer: Tokenizer
    type_tokenizer: Tokenizer

    def __init__(self) -> None:

        self.config = get_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.term_tokenizer = get_tokenizer("term")
        self.type_tokenizer = get_tokenizer("type")

        self.transformer = get_model(
            self.config,
            self.type_tokenizer.get_vocab_size(),
            self.term_tokenizer.get_vocab_size(),
            self.type_tokenizer.token_to_id("[PAD]"),
            self.term_tokenizer.token_to_id("[PAD]"),
        ).to(self.device)

        weights = torch.load(
            "./combinators_llm/combinators-llm.bin", map_location=self.device
        )
        self.transformer.load_state_dict(weights)

    def _preprocess_string(self, src_text: str):

        sos_idx = torch.tensor(
            [self.type_tokenizer.token_to_id("[SOS]")], dtype=torch.int64
        )
        eos_idx = torch.tensor(
            [self.type_tokenizer.token_to_id("[EOS]")], dtype=torch.int64
        )
        pad_idx = torch.tensor(
            [self.type_tokenizer.token_to_id("[PAD]")], dtype=torch.int64
        )

        # Prepare the encoder input
        enc_input_tokens = self.type_tokenizer.encode(src_text).ids
        enc_num_padding_tokens = self.config["seq_len"] - len(enc_input_tokens) - 2

        if enc_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        encoder_input = (
            torch.cat(
                [
                    sos_idx,
                    torch.tensor(enc_input_tokens, dtype=torch.int64),
                    eos_idx,
                    torch.tensor([pad_idx] * enc_num_padding_tokens, dtype=torch.int64),
                ]
            )
            .unsqueeze(0)
            .to(self.device)
        )

        encoder_mask = (
            (encoder_input != pad_idx).unsqueeze(1).unsqueeze(1).int().to(self.device)
        )

        return encoder_input, encoder_mask

    def generate(self, type: str) -> str:

        source, source_mask = self._preprocess_string(type)

        tokens = greedy_decode(
            self.transformer,
            source,
            source_mask,
            self.type_tokenizer,
            self.term_tokenizer,
            self.config["seq_len"],
            self.device,
        )

        return (
            self.term_tokenizer.decode(tokens.tolist())
            .replace("[SOS]", "")
            .replace("[EOS]", "")
        )

    def bgenerate(self, types: List[str]) -> List[str]:

        preprocessed = [self._preprocess_string(t) for t in types]
        source = torch.cat([s for s, _ in preprocessed], dim=0).to(self.device)
        source_mask = torch.cat([m for _, m in preprocessed], dim=0).to(self.device)

        tokens = greedy_decode_batch(
            self.transformer,
            source,
            source_mask,
            self.type_tokenizer,
            self.term_tokenizer,
            self.config["seq_len"],
            self.device,
        )

        decoded = self.term_tokenizer.decode_batch(tokens.tolist())
        return [d.replace("[SOS]", "").replace("[EOS]", "") for d in decoded]

    def __call__(self, input: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(input, list):
            return self.bgenerate(input)
        else:
            return self.generate(input)
