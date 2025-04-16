import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Optional, List, Dict, Tuple, Union, Callable
from nltk.tokenize import word_tokenize
from Preprocessing.prepos_text import preprocess_dataframe

# === Tokenizer ===
def nltk_tokenizer(text: str) -> List[str]:
    """
    Tokenize the input text using NLTK word_tokenize.

    Args:
        text (str): Raw input text.

    Returns:
        List[str]: List of lowercase tokens.
    """
    return word_tokenize(text.lower())


# Tokenizer lookup map
TOKENIZER_MAP: Dict[str, Callable[[str], List[str]]] = {
    "nltk": nltk_tokenizer
}


class SimpleLSTMModel2(nn.Module):
    def __init__(self, weights: Optional[str] = None):
        """
        Initialize the LSTM model, optionally loading from pre-trained weights.

        Args:
            weights (str, optional): Path to saved model checkpoint. Defaults to None.
        """
        super(SimpleLSTMModel2, self).__init__()
        self.weights = weights
        self.embedding = None
        self.lstm = None
        self.fc = None
        self.vocab = None
        self.num_class = None
        self.classes_names = None
        self.tokenizer_name = None
        self.tokenizer = None

        if self.weights is not None:
            checkpoint = torch.load(self.weights, map_location=torch.device('cpu'))

            # Build model with parameters from the checkpoint
            self._build_model(
                vocab_size=len(checkpoint['vocab']),
                output_size=checkpoint['num_class'],
                embed_size=checkpoint.get('embed_size', 256),
                hidden_size=checkpoint.get('hidden_size', 128),
                num_layers=checkpoint.get('num_layers', 2),
                dropout=checkpoint.get('dropout', 0.5),
            )

            self.load_state_dict(checkpoint['model_state_dict'])

            self.vocab = checkpoint['vocab']
            self.num_class = checkpoint['num_class']
            self.classes_names = checkpoint['classes_names']
            self.tokenizer_name = checkpoint.get('tokenizer', 'nltk')
            self.tokenizer = TOKENIZER_MAP.get(self.tokenizer_name)

    def _build_model(
        self,
        vocab_size: int,
        output_size: int,
        embed_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float
    ):
        """
        Build the LSTM model architecture.
        Args:
            vocab_size (int): Size of the vocabulary.
            output_size (int): Number of output classes.
            embed_size (int): Embedding dimension.
            hidden_size (int): LSTM hidden layer size.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate.

        this function is not implemented in the base class.
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM model.
        Args:
            x (torch.Tensor): Input tensor.
        
        This function is not implemented in the base class.
        """
        pass

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict the label of a given text input.

        Args:
            text (str): Input text.

        Returns:
            Tuple[str, float]: Predicted label and its confidence score.
        """
        assert self.weights is not None, "Weights must be loaded before prediction."
        assert self.vocab is not None, "Vocabulary is not available."
        assert self.tokenizer is not None, "Tokenizer is not available."

        self.eval()
        with torch.no_grad():
            # Convert text to token indices (Preprocess)
            text = preprocess_dataframe(text)
            text = text['processed_text'].values[0]
            tokenized = [self.vocab[token] for token in self.tokenizer(text) if token in self.vocab]
            tensor = torch.tensor(tokenized).unsqueeze(0)  # Add batch dimension

            '''Implement Forward Propagation'''
            output = self.forward(tensor)
            probs = F.softmax(output, dim=1)
            predicted_idx = torch.argmax(probs, dim=1).item()
            label = self.classes_names[predicted_idx]
            confidence = probs[0][predicted_idx].item()
            return predicted_idx, label, confidence
    
    def valid(self, dataloader) -> Dict[str, float]:
        """
        Validate the model on the given dataloader and return performance metrics.

        Args:
            dataloader: Dataloader for validation/test data.

        Returns:
            Dict[str, float]: Dictionary containing loss, accuracy, precision, recall, and F1-score.
        """
        assert self.weights is not None, "Weights must be loaded before prediction."

        self.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for _, inputs, labels in dataloader:
                outputs = self.forward(inputs)

                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                all_preds.extend(predicted.tolist())
                all_targets.extend(labels.tolist())

        acc = total_correct / total_samples * 100
        precision, recall, f1 = self.calculate_overall_metrics(all_preds, all_targets)

        print(
            f"Validation | Acc: {acc:.2f}% | "
            f"F1: {f1:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}"
        )
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    def train_model(
        self,
        train_dataloader,
        val_dataloader,
        labels: Dict,
        criterion: nn.Module,
        optimizer: Union[str, torch.optim.Optimizer],
        model_path: str,
        num_epochs: int = 10,
        vocab: Optional[Dict[str, int]] = None,
        embed_size: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5,
        tokenizer_name: str = "nltk"
    ):
        """
        Train the LSTM model and save checkpoints.

        Args:
            train_dataloader: Dataloader for training data.
            val_dataloader: Dataloader for validation data.
            dataset: Full dataset (used to extract class names).
            criterion (nn.Module): Loss function.
            optimizer (str or Optimizer): Optimizer or name of optimizer.
            model_path (str): Directory to save checkpoints.
            num_epochs (int): Number of training epochs.
            vocab (dict): Vocabulary mapping words to indices.
            output_size (int): Number of output classes.
            embed_size (int): Embedding dimension.
            hidden_size (int): LSTM hidden layer size.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate.
            tokenizer_name (str): Name of tokenizer to use.
        """
        os.makedirs(model_path, exist_ok=True)
        best_val_acc = 0.0

        self.vocab = vocab
        self.classes_names = labels
        self.num_class = len(labels)
        output_size = len(labels)

        #Build model
        self._build_model(len(vocab), output_size, embed_size, hidden_size, num_layers, dropout)

        # Initialize optimizer
        if isinstance(optimizer, str):
            if optimizer.lower() == 'adam':
                optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
            elif optimizer.lower() == 'sgd':
                optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
            else:
                raise ValueError(f"Unsupported optimizer: {optimizer}")

        # Train and validate the model based on the provided dataloaders
        for epoch in range(num_epochs):
            train_loss, train_correct, train_total = 0.0, 0, 0
            self.train()

            # Training loop, Forward Propagation and Backpropagation
            for _, inputs, labels in train_dataloader:
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Calculate accuracy and loss
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
                train_loss += loss.item()

            train_acc = train_correct / train_total * 100
            train_loss_avg = train_loss / len(train_dataloader)

            # Validation loop
            # Forward Propagation
            self.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                all_preds, all_targets = [], []
                for _, inputs, labels in val_dataloader:
                    outputs = self.forward(inputs)
                    loss = criterion(outputs, labels)
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
                    val_loss += loss.item()
                    all_preds.extend(predicted.tolist())
                    all_targets.extend(labels.tolist())
            val_precision, val_recall, val_f1 = self.calculate_overall_metrics(all_preds, all_targets)
            val_acc = val_correct / val_total * 100
            val_loss_avg = val_loss / len(val_dataloader)

            # Print log dengan metrik
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss_avg:.4f}, Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss_avg:.4f}, Acc: {val_acc:.2f}% | "
                f"F1: {val_f1:.4f}, Prec: {val_precision:.4f}, Rec: {val_recall:.4f}"
            )
            # Save last checkpoint
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss_avg,
                'train_acc': train_acc,
                'val_loss': val_loss_avg,
                'val_acc': val_acc,
                'vocab': self.vocab,
                'num_class': self.num_class,
                'classes_names': self.classes_names,
                'embed_size': embed_size,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'dropout': dropout,
                'tokenizer': tokenizer_name
            }

            torch.save(save_dict, f'{model_path}/last.pt')

            # Save best model based on validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(save_dict, f'{model_path}/best.pt')

    def calculate_overall_metrics(self , preds: List[int], targets: List[int]) -> Tuple[float, float, float]:
        """
        Calculate overall precision, recall, and F1-score.

        Args:
            preds (List[int]): Predicted class indices.
            targets (List[int]): Ground truth class indices.

        Returns:
            Tuple[float, float, float]: Precision, recall, and F1-score (macro-averaged).
        """
        from collections import defaultdict

        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)
        label_set = set(targets)

        for p, t in zip(preds, targets):
            if p == t:
                tp[t] += 1
            else:
                fp[p] += 1
                fn[t] += 1

        precisions, recalls, f1s = [], [], []

        for label in label_set:
            precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0.0
            recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        avg_precision = sum(precisions) / len(label_set)
        avg_recall = sum(recalls) / len(label_set)
        avg_f1 = sum(f1s) / len(label_set)

        return avg_precision, avg_recall, avg_f1
