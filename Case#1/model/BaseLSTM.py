import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Optional, List, Dict, Tuple, Union, Callable
from nltk.tokenize import word_tokenize
from Preprocessing.prepos_text import preprocess_dataframe
import csv
import yaml
from collections import defaultdict
from Preprocessing.split import split_dataset
from dataloader.dataloader import build_dataloader
from util import yield_tokens, vocabularize, nltk_tokenizer
from tqdm import tqdm


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

        # Read weights file and resume variables
        if self.weights is not None:
            checkpoint = torch.load(self.weights, map_location=torch.device('cpu'))
            print(checkpoint.keys())
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
            print(self.classes_names)
            self.tokenizer_name = checkpoint['tokenizer']
            self.tokenizer = checkpoint['tokenizer']

    def _read_yaml_file(self, yaml_path: str):
        """
        Reads a YAML file and initializes class attributes.

        Args:
            yaml_path (str): Path to the YAML file.
        """
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        self.config = config
        self.dataset_path = config.get("dataset")
        self.test_size = config.get("split", {}).get("test_size", 0.2)
        self.stratify = config.get("split", {}).get("stratify", False)

        preprocess = config.get("preprocess", {})
        self.remove_special_char = preprocess.get("remove_special_char", False)
        self.remove_stopwords = preprocess.get("remove_stopwords", False)
        self.apply_steaming = preprocess.get("apply_steaming", False)
        self.remove_URL = preprocess.get("remove_URL", False)
        self.remove_numbers = preprocess.get("numbers", False)
        self.remove_symbols = preprocess.get("symbols", False)  # Fixed typo
        self.save_prepross_csv = preprocess.get("save_csv", False)

        training = config.get("training_params", {})
        self.optimizer_type = training.get("optimizer", "adam")
        self.model_path = training.get("model_path", "DL_models")
        self.num_epochs = training.get("epochs", 10)
        self.embed_size = training.get("embed_size", 128)
        self.hidden_size = training.get("hidden_size", 128)
        self.num_layers = training.get("num_layers", 1)  # Fixed typo
        self.dropout = training.get("dropout", 0.5)      # Fixed typo
        self.tokenizer_type = training.get("tokenizer", "nltk")
        self.batch_size = training.get("batch_size", 32)

    def _build_dataloader(self):
        """
        Build train and validation dataloaders after preprocessing the dataset.

        Args:
            None (uses class attributes set by the YAML configuration file)
        """
        # Preprocess the dataset
        print("Preprocessing the dataset...")
        self.dataset = preprocess_dataframe(
            df=self.dataset_path,
            remove_stopword=self.remove_stopwords,
            apply_stemming=self.apply_steaming,
            URL=self.remove_URL,
            numbers=self.remove_numbers,
            symbols=self.remove_symbols,
            save_csv=self.save_prepross_csv
        )

        # Encode labels
        self.label_dict = {label: idx for idx, label in enumerate(self.dataset['Sentiment'].unique())}
        self.labels = self.dataset['Sentiment'].map(self.label_dict).fillna(-1).astype(int).values
        self.label_dict = {idx: label for label, idx in self.label_dict.items()}

        print(f"encoded label : {self.label_dict}")
        # Split dataset into train and test
        train_texts, test_texts, train_labels, test_labels = split_dataset(
            texts=self.dataset['processed_text'].values,
            labels=self.labels,
            test_size=self.test_size,
            Stratify=self.stratify
        )

        # Build vocabulary
        self.vocab = vocabularize(texts=self.dataset['processed_text'].values)
        # Create dataloaders
        self.tokenizer = TOKENIZER_MAP[self.tokenizer_type]
        self.train_dataloader = build_dataloader(train_texts, train_labels, self.vocab, self.tokenizer, self.batch_size)
        self.val_dataloader = build_dataloader(test_texts, test_labels, self.vocab, self.tokenizer, self.batch_size)


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
    
    def valid(self, data:str) -> Dict[str, float]:
        """
        Validate the model on the given dataloader and return performance metrics.

        Args:
            dataloader: Dataloader for validation/test data.

        Returns:
            Dict[str, float]: Dictionary containing loss, accuracy, precision, recall, and F1-score.
        """
        assert self.weights is not None, "Weights must be loaded before prediction."
        #read yaml file
        self._read_yaml_file(yaml_path=data)

        # Define the dataloader
        self._build_dataloader()

        # Initialize parameters
        self.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds, all_targets = [], []

        #valid loop
        with torch.no_grad():
            for _, inputs, labels in self.val_dataloader:
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
        data: str
    ):
        """
        Train the LSTM model using the provided configuration file.

        Args:
            data (str): Path to the YAML configuration file.
        """
        # Read YAML file
        self._read_yaml_file(yaml_path=data)

        # Define the dataloader
        self._build_dataloader()

        # Create folder to save the model (avoid overwriting older models)
        if os.path.exists(self.model_path):
            print("Folder already exists. Create a new one.")
            return
        
        os.makedirs(self.model_path, exist_ok=True)
        best_val_acc = 0.0

        # Define/override training variables
        self.vocab = self.vocab
        self.classes_names = self.label_dict
        self.num_class = len(self.label_dict)
        output_size = len(self.label_dict)

        # Build the model based on parameters
        self._build_model(len(self.vocab), output_size, self.embed_size, self.hidden_size, self.num_layers, self.dropout)

        # Build the optimizer
        if isinstance(self.optimizer_type, str):
            if self.optimizer_type.lower() == 'adam':
                optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
            elif self.optimizer_type.lower() == 'sgd':
                optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
            else:
                raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")

        # Define criterion
        criterion = nn.CrossEntropyLoss()

        # === CSV LOGGING: Create or open the CSV file ===
        log_file_path = os.path.join(self.model_path, "training_log.csv")
        with open(log_file_path, mode='w', newline='') as log_file:
            writer = csv.writer(log_file)
            writer.writerow([
                'epoch', 'train_loss', 'train_acc', 
                'val_loss', 'val_acc', 'val_precision', 'val_recall', 'val_f1'
            ])

        # Training loop
        for epoch in range(self.num_epochs):
            train_loss, train_correct, train_total = 0.0, 0, 0
            self.train()

            with tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Training]") as train_bar:
                for _, inputs, labels in train_bar:
                    optimizer.zero_grad()
                    outputs = self.forward(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    _, predicted = torch.max(outputs, 1)
                    train_correct += (predicted == labels).sum().item()
                    train_total += labels.size(0)
                    train_loss += loss.item()

                    train_bar.set_postfix({
                        "Loss": f"{train_loss / (train_total / labels.size(0)):.4f}",
                        "Acc": f"{train_correct / train_total * 100:.2f}%"
                    })

            train_acc = train_correct / train_total * 100
            train_loss_avg = train_loss / len(self.train_dataloader)

            self.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            all_preds, all_targets = [], []
            with torch.no_grad():
                with tqdm(self.val_dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Validation]") as val_bar:
                    for _, inputs, labels in val_bar:
                        outputs = self.forward(inputs)
                        loss = criterion(outputs, labels)
                        _, predicted = torch.max(outputs, 1)
                        val_correct += (predicted == labels).sum().item()
                        val_total += labels.size(0)
                        val_loss += loss.item()
                        all_preds.extend(predicted.tolist())
                        all_targets.extend(labels.tolist())

                        val_bar.set_postfix({
                            "Loss": f"{val_loss / (val_total / labels.size(0)):.4f}",
                            "Acc": f"{val_correct / val_total * 100:.2f}%"
                        })

            val_precision, val_recall, val_f1 = self.calculate_overall_metrics(all_preds, all_targets)
            val_acc = val_correct / val_total * 100
            val_loss_avg = val_loss / len(self.val_dataloader)

            # === CSV LOGGING: Append training results ===
            with open(log_file_path, mode='a', newline='') as log_file:
                writer = csv.writer(log_file)
                writer.writerow([
                    epoch + 1, train_loss_avg, train_acc,
                    val_loss_avg, val_acc, val_precision, val_recall, val_f1
                ])

            print(
                f"Epoch {epoch+1}/{self.num_epochs} | "
                f"Train Loss: {train_loss_avg:.4f}, Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss_avg:.4f}, Acc: {val_acc:.2f}% | "
                f"F1: {val_f1:.4f}, Prec: {val_precision:.4f}, Rec: {val_recall:.4f}"
            )

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
                'embed_size': self.embed_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'tokenizer': self.tokenizer
            }

            torch.save(save_dict, f'{self.model_path}/last.pt')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(save_dict, f'{self.model_path}/best.pt')

    def calculate_overall_metrics(self , preds: List[int], targets: List[int]) -> Tuple[float, float, float]:
        """
        Calculate overall precision, recall, and F1-score.

        Args:
            preds (List[int]): Predicted class indices.
            targets (List[int]): Ground truth class indices.

        Returns:
            Tuple[float, float, float]: Precision, recall, and F1-score (macro-averaged).
        """
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
