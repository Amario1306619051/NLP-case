from nltk.tokenize import word_tokenize
from typing import List, Iterator
from torchtext.vocab import Vocab, build_vocab_from_iterator

# === Tokenizer dengan NLTK ===
def nltk_tokenizer(text: str):
    """
    Tokenizes a single text input into lowercase words using NLTK tokenizer.

    Args:
        text (str): Input text string.

    Returns:
        List[str]: List of tokenized words.
    """
    return word_tokenize(text.lower())

# === Fungsi untuk yield token dari kumpulan teks ===
def yield_tokens(texts: List[str]) -> Iterator[List[str]]:
    """
    Generator that yields tokenized words for each text in the list.
    Useful for feeding into vocab builder.

    Args:
        texts (List[str]): List of raw text strings.

    Yields:
        List[str]: Tokenized word list for each text.
    """
    for text in texts:
        yield nltk_tokenizer(text)

# === Build vocab ===
def vocabularize(texts: List[str]):
    """
    Build a vocabulary object from a list of texts.

    Args:
        texts (List[str]): List of raw texts.

    Returns:
        Vocab: TorchText Vocab object with <unk> and <pad> tokens.
    """
    vocab = build_vocab_from_iterator(yield_tokens(texts), specials=["<unk>", "<pad>"])
    vocab.set_default_index(vocab["<unk>"])
    return vocab