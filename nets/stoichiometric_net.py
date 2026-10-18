from torch import nn

class StoichiometricNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StoichiometricNet, self).__init__()

        self.hidden_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Softplus()  # Ensure non-negative outputs
        )

    def forward(self, x):
        hidden_output = self.hidden_layer(x)
        output = self.output_layer(hidden_output)
        return output

