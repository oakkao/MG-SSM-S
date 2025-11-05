import torch
import torch.nn as nn

class CustomMambaCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomMambaCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # cell input gate
        self.W_ic = nn.Linear(input_size, hidden_size)
        self.W_hc = nn.Linear(hidden_size, hidden_size)
        self.b_c = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, hidden_state):
        h_prev = hidden_state

        # Hidden state
        h_t = self.W_ic(x) + self.W_hc(h_prev) + self.b_c

        return h_t

class CustomMambaModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Initializes the CustomMambaModel.

        Args:
            input_size (int): The number of expected features in the input $x$.
            hidden_size (int): The number of features in the hidden state $h$.
            num_layers (int): Number of recurrent layers.
            output_size (int): The size of the output from the final linear layer.
        """
        super(CustomMambaModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm_cells = nn.ModuleList([CustomMambaCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        batch_size, seq_len, _ = x.size()
        hidden_state = [torch.zeros(batch_size, self.hidden_size).to(x.device) # Initialize with a single tensor
                         for _ in range(self.num_layers)]

        for t in range(seq_len):
            input_t = x[:, t, :]
            for i in range(self.num_layers):
                h_prev = hidden_state[i]

                # Mamba part (using the Custom_MambaCell)
                h_t = self.lstm_cells[i](input_t, h_prev) # Pass a single tensor
                hidden_state[i] = h_t
                input_t = h_t # Use the output of the current layer as input for the next layer

        # Decode the hidden state of the last time step of the last layer
        last_hidden_state = hidden_state[-1]
        out = self.fc(last_hidden_state)
        return out