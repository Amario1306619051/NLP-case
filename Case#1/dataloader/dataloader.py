import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import nltk
from typing import List, Tuple, Callable

# === Dataset Class ===
class TextDataset(Dataset):
    """
    PyTorch Dataset for tokenized text classification.

    Attributes:
        texts (List[str]): List of raw text inputs.
        labels (List[int]): Corresponding list of class labels.
        vocab (Vocab): Vocabulary object that maps tokens to indices.
        tokenizer (Callable[[str], List[str]]): Tokenizer function.
    """
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        vocab,
        tokenizer: Callable[[str], List[str]]
    ):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """
        Returns the tokenized and encoded input and label for a given index.

        Args:
            idx (int): Index of the data point.

        Returns:
            Tuple[str, Tensor, Tensor]: Raw text, token IDs, and label tensor.
        """
        raw_text = self.texts[idx]
        tokens = self.tokenizer(raw_text)
        token_ids = [self.vocab[token] for token in tokens]
        return raw_text, torch.tensor(token_ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# === Build DataLoader ===
def build_dataloader(
    texts: List[str],
    labels: List[int],
    vocab,
    tokenizer: Callable[[str], List[str]],
    batch_size: int = 32
) -> DataLoader:
    """
    Builds a DataLoader for training/validation with padding and batching.

    Args:
        texts (List[str]): List of raw text inputs.
        labels (List[int]): Corresponding list of class labels.
        vocab (Vocab): Vocabulary object.
        tokenizer (Callable): Tokenizer function.
        batch_size (int, optional): Batch size. Defaults to 32.

    Returns:
        DataLoader: PyTorch DataLoader with padded sequences.
    """
    dataset = TextDataset(texts, labels, vocab, tokenizer)

    def collate_fn(batch: List[Tuple[str, torch.Tensor, torch.Tensor]]) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Custom collate function to pad sequences and batch data.

        Args:
            batch (List[Tuple]): List of tuples (raw_text, token_ids, label).

        Returns:
            Tuple[List[str], Tensor, Tensor]: Raw texts, padded input IDs, labels.
        """
        raw_texts, token_lists, label_list = zip(*batch)
        padded_token_ids = pad_sequence(token_lists, batch_first=True, padding_value=vocab["<pad>"])
        labels = torch.tensor(label_list, dtype=torch.long)
        return list(raw_texts), padded_token_ids, labels

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader
