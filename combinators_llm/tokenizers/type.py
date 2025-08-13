from tokenizers import Tokenizer, Regex
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split

vocab = {
    "[UNK]": 0,
    "[PAD]": 1,
    "[SOS]": 2,
    "[EOS]": 3,
    "->": 4,
    "(": 5,
    ")": 6,
} | { chr(65 + i) : i + 7 for i in range(26) }

if __name__ == "__main__" :
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Split(Regex("([A-Z]|->|\(|\))"), behavior="removed", invert=True)
    
    tokenizer.save("combinators_llm/tokenizers/type-tokenizer.json")