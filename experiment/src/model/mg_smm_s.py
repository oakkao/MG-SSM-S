import torch
import torch.nn as nn
import numpy as np

class MgSmmSCell(nn.Module):
    def __init__(self, input_size, hidden_size, gate_size):
        super(MgSmmSCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate_size = gate_size

        # linear input gate
        self.W_A = nn.Linear(hidden_size, hidden_size)
        self.W_B = nn.Linear(input_size, hidden_size)
        self.W_bh = nn.Parameter(torch.zeros(hidden_size))

        # multiplicative input gate
        self.W_E = nn.Linear(gate_size, gate_size)
        self.W_F = nn.Linear(input_size, gate_size)
        self.W_bg = nn.Parameter(torch.zeros(gate_size))

    def forward(self, x, hidden_state):
        h_prev, g_prev = hidden_state

        # Hidden state
        h_t =  self.W_A(h_prev) + self.W_B(x) +  self.W_bh
        g_t =  self.W_E(g_prev) * torch.repeat_interleave(x, self.gate_size, dim=1) + self.W_F(x) + self.W_bg

        return h_t, g_t

class MgSmmSModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, gate_size = 0):
        """
        Initializes the CustomMambaModel.

        Args:
            input_size (int): The number of expected features in the input $x$.
            hidden_size (int): The number of features in the hidden state $h$.
            num_layers (int): Number of recurrent layers.
            output_size (int): The size of the output from the final linear layer.
            gate_size (int): The number of features in the multiplicative gate cell.
        """
        super(MgSmmSModel, self).__init__()
        self.hidden_size = hidden_size
        if gate_size == 0:
            gate_size = int(hidden_size/2)
        self.gate_size = gate_size

        self.num_layers = num_layers
        self.SSM_blocks = nn.ModuleList([MgSmmSCell(input_size if i == 0 else hidden_size, hidden_size, gate_size) for i in range(num_layers)])

        self.W_C = nn.Linear(hidden_size, output_size)
        self.W_D = nn.Linear(input_size, output_size)
        self.W_J = nn.Linear(gate_size, output_size)

    def forward(self, x):
        """
        Performs the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        batch_size, seq_len, _ = x.size()
        hidden_state = [(torch.zeros(batch_size, self.hidden_size).to(x.device),
                                torch.ones(batch_size, self.gate_size).to(x.device)
                         ) for _ in range(self.num_layers)]

        for t in range(seq_len):
            input_t = x[:, t, :]
            for i in range(self.num_layers):
                h_prev, g_prev = hidden_state[i]

                # Mamba part (using the Custom_MambaCell)
                h_t, c_t = self.SSM_blocks[i](input_t, (h_prev, g_prev)) # Pass a single tensor
                hidden_state[i] = (h_t, g_prev)
                input_t = h_t # Use the output of the current layer as input for the next layer

        # Decode the hidden state of the last time step of the last layer
        h_prev, g_prev = hidden_state[-1]

        out = self.W_C(h_prev) + self.W_D(x[:, seq_len - 1, :])+ self.W_J(g_prev)
        return out