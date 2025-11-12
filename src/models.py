"""
RNN model architectures for sentiment classification.
Supports: RNN, LSTM, Bidirectional LSTM with configurable activations.
"""
import torch
import torch.nn as nn


class SentimentRNN(nn.Module):
    """
    RNN-based sentiment classifier for binary classification.
    
    Architecture:
    - Embedding layer (size: 100)
    - RNN/LSTM layer (configurable)
    - 2 fully connected hidden layers (size: 64 each)
    - Dropout (0.3-0.5)
    - Output layer with sigmoid activation
    """
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_size=64, num_layers=2,
                 rnn_type='RNN', activation='relu', dropout=0.3, bidirectional=False):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension (default: 100)
            hidden_size: Hidden layer size (default: 64)
            num_layers: Number of RNN layers (default: 2)
            rnn_type: 'RNN' or 'LSTM'
            activation: 'relu', 'tanh', or 'sigmoid' (applied in FC layers)
            dropout: Dropout rate (default: 0.3)
            bidirectional: Whether to use bidirectional RNN
        """
        super(SentimentRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        rnn_type = rnn_type.upper()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layer
        # Note: PyTorch RNN only supports 'tanh' or 'relu' for nonlinearity
        if rnn_type == 'RNN':
            rnn_activation = 'tanh' if activation.lower() == 'sigmoid' else activation.lower()
            if rnn_activation not in ['tanh', 'relu']:
                rnn_activation = 'tanh'
            self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers,
                             batch_first=True,
                             dropout=dropout if num_layers > 1 else 0,
                             bidirectional=bidirectional,
                             nonlinearity=rnn_activation)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers,
                              batch_first=True,
                              dropout=dropout if num_layers > 1 else 0,
                              bidirectional=bidirectional)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}. Use 'RNN' or 'LSTM'")
        
        # Determine RNN output size (doubled if bidirectional)
        rnn_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Fully connected layers (2 hidden layers as per requirements)
        self.fc1 = nn.Linear(rnn_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Activation function for FC layers
        activation_lower = activation.lower()
        if activation_lower == 'relu':
            self.activation = nn.ReLU()
        elif activation_lower == 'tanh':
            self.activation = nn.Tanh()
        elif activation_lower == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layer (binary classification with sigmoid)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
        
        Returns:
            Output tensor of shape (batch_size, 1) with sigmoid activation
        """
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # RNN
        rnn_out, _ = self.rnn(embedded)
        
        # Get final output (last timestep)
        if self.bidirectional:
            # Concatenate forward and backward final outputs
            forward_final = rnn_out[:, -1, :self.hidden_size]
            backward_final = rnn_out[:, 0, self.hidden_size:]
            rnn_final = torch.cat([forward_final, backward_final], dim=1)
        else:
            rnn_final = rnn_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Fully connected layers with activation and dropout
        out = self.fc1(rnn_final)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        # Output layer with sigmoid
        out = self.fc_out(out)
        out = self.sigmoid(out)
        
        return out
