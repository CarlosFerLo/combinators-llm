from tokenizers import Tokenizer, Regex
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split, Sequence

# Allowed tokens
vocab = {"[UNK]": 0, "[PAD]": 1, "[SOS]": 2, "[EOS]": 3, "S": 4, "K": 5, "(": 6, ")": 7}

if __name__ == "__main__":
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Sequence(  # type: ignore
        [Split(Regex(r"\s"), behavior="removed"), Split("", behavior="removed")]
    )

    tokenizer.save("combinators_llm/tokenizers/term-tokenizer.json")
