import torch
import torch.nn as nn

class OriginalLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Initializes the OriginalLSTMModel.

        Args:
            input_size (int): The number of expected features in the input $x$.
            hidden_size (int): The number of features in the hidden state $h$.
            num_layers (int): Number of recurrent layers.
            output_size (int): The size of the output from the final linear layer.
        """
        super(OriginalLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # The output of a LSTM at each time step
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
    
class BidirectionalLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Initializes the BidirectionalLSTMModel.

        Args:
            input_size (int): The number of expected features in the input $x$.
            hidden_size (int): The number of features in the hidden state $h$.
            num_layers (int): Number of recurrent layers.
            output_size (int): The size of the output from the final linear layer.
        """
        super(BidirectionalLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # The output of a bidirectional LSTM is the concatenation of the forward and backward hidden states
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Initialize hidden and cell states
        # For bidirectional LSTM, the shape is (2 * num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, 2 * hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
    
class UnidirectionalGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Initializes the UnidirectionalGRUModel.

        Args:
            input_size (int): The number of expected features in the input $x$.
            hidden_size (int): The number of features in the hidden state $h$.
            num_layers (int): Number of recurrent layers.
            output_size (int): The size of the output from the final linear layer.
        """
        super(UnidirectionalGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        # The output of a unidirectional GRU at each time step is just the hidden_size
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Initialize the hidden state
        # For a unidirectional GRU, the shape is (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate GRU
        out, _ = self.gru(x, h0) # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step.
        out = self.fc(out[:, -1, :])
        return out
