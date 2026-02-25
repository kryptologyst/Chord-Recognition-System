"""Neural network models for chord recognition."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class CRNNChordModel(nn.Module):
    """Convolutional Recurrent Neural Network for chord recognition."""
    
    def __init__(
        self,
        input_size: int = 12,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 25,
        dropout: float = 0.3,
        bidirectional: bool = True,
        cnn_layers: int = 3,
        cnn_channels: List[int] = [32, 64, 128],
        kernel_sizes: List[int] = [3, 3, 3],
        pool_sizes: List[int] = [2, 2, 2],
        rnn_type: str = "lstm",
    ):
        """Initialize CRNN model.
        
        Args:
            input_size: Input feature size (chroma = 12).
            hidden_size: Hidden size for RNN layers.
            num_layers: Number of RNN layers.
            num_classes: Number of chord classes.
            dropout: Dropout rate.
            bidirectional: Use bidirectional RNN.
            cnn_layers: Number of CNN layers.
            cnn_channels: Number of channels for each CNN layer.
            kernel_sizes: Kernel sizes for CNN layers.
            pool_sizes: Pool sizes for CNN layers.
            rnn_type: Type of RNN ('lstm' or 'gru').
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.lower()
        
        # CNN layers
        self.cnn_layers = nn.ModuleList()
        in_channels = 1  # Input is mono audio features
        
        for i in range(cnn_layers):
            out_channels = cnn_channels[i]
            kernel_size = kernel_sizes[i]
            pool_size = pool_sizes[i]
            
            self.cnn_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(pool_size),
                nn.Dropout(dropout)
            ))
            in_channels = out_channels
        
        # Calculate CNN output size
        cnn_output_size = cnn_channels[-1]
        
        # RNN layers
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=cnn_output_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(
                input_size=cnn_output_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        # Output layer
        rnn_output_size = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(rnn_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).
            
        Returns:
            Output tensor of shape (batch_size, seq_len, num_classes).
        """
        batch_size, seq_len, _ = x.shape
        
        # Reshape for CNN: (batch_size, 1, seq_len * input_size)
        x = x.view(batch_size, 1, -1)
        
        # Apply CNN layers
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
        
        # Reshape for RNN: (batch_size, seq_len, cnn_output_size)
        cnn_output_size = x.size(1)
        x = x.view(batch_size, seq_len, cnn_output_size)
        
        # Apply RNN
        x, _ = self.rnn(x)
        
        # Apply classifier
        x = self.classifier(x)
        
        return x


class TransformerChordModel(nn.Module):
    """Transformer model for chord recognition."""
    
    def __init__(
        self,
        input_size: int = 12,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "relu",
        num_classes: int = 25,
        max_seq_length: int = 1000,
    ):
        """Initialize Transformer model.
        
        Args:
            input_size: Input feature size (chroma = 12).
            d_model: Model dimension.
            nhead: Number of attention heads.
            num_encoder_layers: Number of encoder layers.
            num_decoder_layers: Number of decoder layers.
            dim_feedforward: Feedforward dimension.
            dropout: Dropout rate.
            activation: Activation function.
            num_classes: Number of chord classes.
            max_seq_length: Maximum sequence length.
        """
        super().__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.num_classes = num_classes
        self.max_seq_length = max_seq_length
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # Output layer
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).
            mask: Attention mask.
            
        Returns:
            Output tensor of shape (batch_size, seq_len, num_classes).
        """
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Apply classifier
        x = self.classifier(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension.
            dropout: Dropout rate.
            max_len: Maximum sequence length.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            
        Returns:
            Tensor with positional encoding added.
        """
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class SimpleChordModel(nn.Module):
    """Simple feedforward model for chord recognition."""
    
    def __init__(
        self,
        input_size: int = 12,
        hidden_sizes: List[int] = [128, 64],
        num_classes: int = 25,
        dropout: float = 0.3,
        activation: str = "relu",
    ):
        """Initialize simple model.
        
        Args:
            input_size: Input feature size (chroma = 12).
            hidden_sizes: List of hidden layer sizes.
            num_classes: Number of chord classes.
            dropout: Dropout rate.
            activation: Activation function.
        """
        super().__init__()
        
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU() if activation == "relu" else nn.Tanh(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_size) or (batch_size, seq_len, input_size).
            
        Returns:
            Output tensor of shape (batch_size, num_classes) or (batch_size, seq_len, num_classes).
        """
        # Handle both 2D and 3D inputs
        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
            x = x.view(-1, self.input_size)
            x = self.network(x)
            x = x.view(batch_size, seq_len, self.num_classes)
        else:
            x = self.network(x)
        
        return x


def create_model(
    model_type: str,
    input_size: int = 12,
    num_classes: int = 25,
    **kwargs
) -> nn.Module:
    """Create a chord recognition model.
    
    Args:
        model_type: Type of model ('crnn', 'transformer', 'simple').
        input_size: Input feature size.
        num_classes: Number of chord classes.
        **kwargs: Additional model parameters.
        
    Returns:
        Initialized model.
    """
    if model_type.lower() == "crnn":
        return CRNNChordModel(input_size=input_size, num_classes=num_classes, **kwargs)
    elif model_type.lower() == "transformer":
        return TransformerChordModel(input_size=input_size, num_classes=num_classes, **kwargs)
    elif model_type.lower() == "simple":
        return SimpleChordModel(input_size=input_size, num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
