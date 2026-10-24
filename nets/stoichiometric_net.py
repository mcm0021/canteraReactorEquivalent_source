import torch
from torch import nn

from src.stoichiometry import coordinates_to_amounts
from src.state_conversion import amounts_to_state


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

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def predict_real(self, x, n0, Q, Vgas, W, A, Gamma, sizes, T0,
                     reverse_scaling_fn):
        predictions = self.predict(x)
        predictions_real = reverse_scaling_fn(predictions)

        predictions_real = predictions_real.squeeze(0).detach().cpu().numpy()
        z_pred, delta_T = predictions_real[:-1], predictions_real[-1]
        n1_pred = coordinates_to_amounts(n0, z_pred, Q)
        mass_fractions, coverages, gas_density = amounts_to_state(
            n1_pred, Vgas, W, A, Gamma, sizes)
        T_pred = T0 + delta_T
        return T_pred, mass_fractions, coverages, gas_density
