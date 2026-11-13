import torch
from torch import nn

from src.reactor import get_reactor_constants, initialize_reactor, initialize_variables
from src.stoichiometry import build_stoichiometric_matrix, compute_stoichiometric_basis, coordinates_to_amounts
from src.state_conversion import amounts_to_state, state_to_amounts


def coordinates_to_real_state(n0, z_pred, Q, Vgas, W, A, Gamma, sizes):
    n1_pred = coordinates_to_amounts(n0, z_pred, Q)
    return amounts_to_state(n1_pred, Vgas, W, A, Gamma, sizes)


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

        nn.init.uniform_(self.output_layer.weight, -1e-3, 1e-3)

    def forward(self, x):
        h = self.input_layer(x)
        h = h + self.hidden_block(h)
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
        mass_fractions, coverages, gas_density = coordinates_to_real_state(
            n0, z_pred, Q, Vgas, W, A, Gamma, sizes)
        T_pred = T0 + delta_T
        return T_pred, mass_fractions, coverages, gas_density

    def predict_real_from_input(self, single_x, coordinates,
                                reverse_scaling_fn, mechanism_path):
        gas, solid, surface = initialize_variables(mechanism_path)
        num_gas_species = len(gas.species_names)

        S, _ = build_stoichiometric_matrix(gas, surface)
        Q, _, _ = compute_stoichiometric_basis(S)

        T_in = float(single_x[0])
        P_in = float(single_x[1])
        v_in = float(single_x[2])
        cat_area = float(single_x[4])
        gas_y = single_x[5:5 + num_gas_species]
        surf_cov = single_x[5 + num_gas_species:]

        reactor, reactor_surface = initialize_reactor(gas, solid, surface,
                                                      T_in, P_in, v_in,
                                                      cat_area, gas_y,
                                                      surf_cov)
        W, Gamma, sizes, Vgas, A = get_reactor_constants(
            gas, solid, surface, v_in, cat_area, 1.1e-6)
        n0 = state_to_amounts(
            reactor.phase.Y,
            reactor.phase.density,
            Vgas,
            W,
            reactor_surface.coverages,
            A,
            Gamma,
            sizes,
        )

        if coordinates is not None:
            return coordinates_to_real_state(n0, coordinates, Q, Vgas, W, A,
                                             Gamma, sizes)
        else:
            return self.predict_real(
                coordinates,
                n0,
                Q,
                W,
                Vgas,
                A,
                Gamma,
                sizes,
                T_in,
                reverse_scaling_fn,
            )
