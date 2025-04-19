import torch
import torch.nn as nn
from typing import Optional
from model.BaseLSTM import SimpleLSTMModel2


class LSTMModel3(SimpleLSTMModel2):
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
        Build a deeper LSTM model with fixed embedding and LSTM size.

        Args:
            vocab_size (int): Vocabulary size.
            output_size (int): Number of output classes.
            embed_size (int): Ignored (fixed to 512 here).
            hidden_size (int): Ignored (fixed to 128 here).
            num_layers (int): Ignored (fixed to 4 here).
            dropout (float): Ignored (fixed to 0.5 here).
        """
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for LSTMModel3.

        Args:
            x (Tensor): Input tensor of token indices.

        Returns:
            Tensor: Logits for each class.
        """
        embedded = self.embedding(x)
        lstm_out, (hn, _) = self.lstm(embedded)
        out = self.fc(hn[-1])  # Use the last hidden state
        return out


class BidirectionalLSTMModel(SimpleLSTMModel2):
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
        Build a bidirectional LSTM model.

        Args:
            vocab_size (int): Vocabulary size.
            output_size (int): Number of output classes.
            embed_size (int): Ignored (fixed to 512 here).
            hidden_size (int): Ignored (fixed to 128 here).
            num_layers (int): Ignored (fixed to 4 here).
            dropout (float): Ignored (fixed to 0.5 here).
        """
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            embed_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True
        )
        self.fc = nn.Linear(hidden_size*2, output_size)  # times 2 for bidirectional output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Bidirectional LSTM.

        Args:
            x (Tensor): Input tensor of token indices.

        Returns:
            Tensor: Logits for each class.
        """
        embedded = self.embedding(x)
        lstm_out, (hn, _) = self.lstm(embedded)
        # Concatenate last hidden states from both directions
        hn_bi = torch.cat((hn[-2], hn[-1]), dim=1)
        out = self.fc(hn_bi)
        return out


class GRUModel(SimpleLSTMModel2):
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
        Build a GRU-based sequence model.

        Args:
            vocab_size (int): Vocabulary size.
            output_size (int): Number of output classes.
            embed_size (int): Embedding dimension.
            hidden_size (int): GRU hidden size.
            num_layers (int): Number of GRU layers.
            dropout (float): Dropout rate.
        """
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for GRU model.

        Args:
            x (Tensor): Input tensor of token indices.

        Returns:
            Tensor: Logits for each class.
        """
        embedded = self.embedding(x)
        gru_out, hn = self.gru(embedded)
        out = self.fc(hn[-1])  # Use last GRU hidden state
        return out


class LSTM_CNN_Model(SimpleLSTMModel2):
    def _build_model(
        self,
        vocab_size: int,
        output_size: int,
        embed_size: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5
    ):
        """
        Build a hybrid LSTM + CNN model.

        Args:
            vocab_size (int): Vocabulary size.
            output_size (int): Number of output classes.
            embed_size (int): Embedding dimension.
            hidden_size (int): LSTM hidden size.
            num_layers (int): Number of LSTM layers.
            dropout (float): Dropout rate.
        """
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for LSTM + CNN model.

        Args:
            x (Tensor): Input tensor of token indices.

        Returns:
            Tensor: Logits for each class.
        """
        embedded = self.embedding(x)                             # (batch_size, seq_len, embed_size)
        lstm_out, _ = self.lstm(embedded)                        # (batch_size, seq_len, hidden_size)
        lstm_out = lstm_out.permute(0, 2, 1)                     # (batch_size, hidden_size, seq_len)
        cnn_out = self.conv(lstm_out)                            # (batch_size, hidden_size, seq_len)
        cnn_out = cnn_out.permute(0, 2, 1)                       # (batch_size, seq_len, hidden_size)
        out = self.fc(cnn_out[:, -1, :])                         # (batch_size, output_size)
        return out
