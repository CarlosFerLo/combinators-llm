from tokenizers import Tokenizer
from pathlib import Path

TOKENIZER_FOLDER = Path("combinators_llm/tokenizers")


def get_tokenizer(name: str) -> Tokenizer:
    tokenizer_path = str(TOKENIZER_FOLDER / f"{name}-tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    return tokenizer
