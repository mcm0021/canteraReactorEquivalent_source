import torch
from torch import nn

from src.stoichiometry import coordinates_to_amounts
from src.state_conversion import amounts_to_state


class StoichiometricNet(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        # SiLU (Swish) is smooth, non-monotonic, and prevents dead neurons
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
        )
        
        # Deeper hidden representation with a residual skip connection
        self.hidden_block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Optional: If your scaled outputs are still very small, initialize 
        # the final layer with smaller weights so initial predictions start near 0
        nn.init.uniform_(self.output_layer.weight, -1e-3, 1e-3)

    def forward(self, x):
        h = self.input_layer(x)
        h = h + self.hidden_block(h)  # Residual connection stabilizes gradients
        return self.output_layer(h)

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