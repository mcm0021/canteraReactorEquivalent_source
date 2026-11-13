from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import torch
from sklearn.metrics import mean_absolute_percentage_error
from torch.utils.data import DataLoader

from datasets.dataset_basic import BasicDataset
from datasets.dataset_sotichiometry import StoichiometryDataset
from nets.stoichiometric_net import StoichiometricNet
from src.reactor import get_reactor_constants, initialize_reactor, initialize_variables
from src.stoichiometry import (
    build_stoichiometric_matrix,
    compute_stoichiometric_basis,
    state_to_amounts,
)
from utils.preprocessing import preprocess_stoichiometry, get_bounds, preprocess_stoichiometry_output

CPU_DEVICE = torch.device("cpu")

EVALUATION_INPUT_PATH = Path("data/100k_samples/input_val.npy")
EVALUATION_OUTPUT_PATH = Path("data/100k_samples/output_val.npy")
ACTUAL_OUTPUT_PATH = Path("data/100k_samples/actual_output_val.npy")
MODEL_PATH = Path("model.pth")
MECHANISM_PATH = Path(
    __file__).parent / "mechanisms" / "coOxidationRhAlumina.yaml"

INPUT_MIN_BOUNDS, INPUT_MAX_BOUNDS = get_bounds()

OUTPUT_MEAN = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
OUTPUT_STD = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)


def reverse_scaling(data: torch.Tensor, c=10e-14) -> torch.Tensor:
    """Reverses the z-score scaling on CPU using pre-allocated mean and std."""
    data = data * OUTPUT_STD + OUTPUT_MEAN
    data[:, :-1] = c * torch.sinh(
        data[:, :-1]
    )  # Reverse the arcsinh transformation for stoichiometric coordinates
    return data


@torch.inference_mode()
def evaluate_batches(
    model: torch.nn.Module,
    loader: DataLoader,
    mape_threshold: float = 0.1,
) -> Tuple[float, int, int]:
    """
    Fast, vectorized batch evaluation directly on the CPU.
    """
    model.eval()
    all_sample_mapes = []
    eps = 1e-8
    for inputs, targets in loader:
        inputs = inputs.to(device=CPU_DEVICE, dtype=torch.float32)
        predictions = model.predict(inputs)

        relative_error = torch.abs(
            (targets - predictions) / torch.clamp(torch.abs(targets), min=eps))
        relative_error = torch.nan_to_num(
            relative_error,
            nan=1.0e6,
            posinf=1.0e6,
            neginf=1.0e6,
        )
        batch_mape = torch.mean(relative_error, dim=1)
        all_sample_mapes.append(batch_mape)

    sample_mapes = torch.cat(all_sample_mapes)
    above_threshold = int((sample_mapes > mape_threshold).sum().item())
    below_threshold = int((sample_mapes <= mape_threshold).sum().item())
    mean_mape = float(sample_mapes.mean().item())

    return mean_mape, above_threshold, below_threshold


@torch.inference_mode()
def evaluate_physical_states(
    model: StoichiometricNet,
    scaled_dataset: StoichiometryDataset,
    raw_dataset: BasicDataset,
    mechanism_path: Path,
    metric_fn: Callable = mean_absolute_percentage_error,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates physical state reconstruction (T, mass fractions, coverages) on CPU.
    """
    model.eval()
    gas, solid, surface = initialize_variables(mechanism_path)
    num_gas_species = len(gas.species_names)

    S, _ = build_stoichiometric_matrix(gas, surface)
    Q, _, _ = compute_stoichiometric_basis(S)

    t_mape_list = []
    mass_frac_mape_list = []
    coverages_mape_list = []
    for (scaled_in, _), (raw_in, actual_target) in zip(scaled_dataset,
                                                       raw_dataset):
        T_in = float(raw_in[0].item())
        P_in = float(raw_in[1].item())
        v_in = float(raw_in[2].item())
        cat_area = float(raw_in[4].item())
        gas_y = raw_in[5:5 + num_gas_species].tolist()
        surf_cov = raw_in[5 + num_gas_species:].tolist()

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

        T_pred, mass_fractions, coverages, _ = model.predict_real(
            scaled_in.unsqueeze(0),
            n0,
            Q,
            Vgas,
            W,
            A,
            Gamma,
            sizes,
            T_in,
            reverse_scaling_fn=reverse_scaling,
        )

        actual_np = actual_target.numpy()
        actual_T = float(actual_np[0])
        pred_T = float(np.squeeze(T_pred))

        t_abs_err = abs(pred_T - actual_T)
        t_mape = t_abs_err / max(abs(actual_T), 1e-8)
        mass_frac_mape = metric_fn(mass_fractions,
                                   actual_np[2:2 + num_gas_species])
        coverages_mape = metric_fn(coverages, actual_np[2 + num_gas_species:])

        t_mape_list.append(t_mape)
        mass_frac_mape_list.append(mass_frac_mape)
        coverages_mape_list.append(coverages_mape)

        print(f"T_pred: {pred_T:.2f} K | "
              f"T Abs Err: {t_abs_err:.2f} K (MAPE: {t_mape:.4f}) | "
              f"Mass Frac MAPE: {mass_frac_mape:.4f} | "
              f"Coverages MAPE: {coverages_mape:.4f}")

    return np.array(t_mape_list), np.array(mass_frac_mape_list), np.array(
        coverages_mape_list)


def mape_buckets(delta_t_mape: np.ndarray, mass_frac_mape: np.ndarray,
                 coverages_mape: np.ndarray) -> None:
    """
    Categorizes MAPE values into buckets and prints the counts for each bucket.
    """
    delta_t_buckets = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    mass_frac_buckets = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    coverages_buckets = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    print("Delta T MAPE Buckets:")
    prev_count = 0
    for threshold in delta_t_buckets:
        count = np.sum(delta_t_mape <= threshold)
        print(f"MAPE <= {threshold:.2f}: {count - prev_count}")
        prev_count = count
    print(f"MAPE >= {delta_t_buckets[-1]:.2f}: {len(delta_t_mape) - prev_count}")

    print("\nMass Fraction MAPE Buckets:")
    prev_count = 0
    for threshold in mass_frac_buckets:
        count = np.sum(mass_frac_mape <= threshold)
        print(f"MAPE <= {threshold:.2f}: {count - prev_count}")
        prev_count = count
    print(f"MAPE >= {mass_frac_buckets[-1]:.2f}: {len(mass_frac_mape) - prev_count}")
    print("\nCoverages MAPE Buckets:")
    prev_count = 0
    for threshold in coverages_buckets:
        count = np.sum(coverages_mape <= threshold)
        print(f"MAPE <= {threshold:.2f}: {count - prev_count}")
        prev_count = count
    print(f"MAPE >= {coverages_buckets[-1]:.2f}: {len(coverages_mape) - prev_count}")


def main() -> None:
    evaluation_data = StoichiometryDataset(
        str(EVALUATION_INPUT_PATH),
        str(EVALUATION_OUTPUT_PATH),
        Train=False,
        transform_input=preprocess_stoichiometry,
        transform_output=preprocess_stoichiometry_output,
        output_scalers=([], []),
        mean=OUTPUT_MEAN,
        std=OUTPUT_STD,
    )
    evaluation_loader = DataLoader(
        evaluation_data,
        batch_size=256,
        shuffle=False,
    )

    model = StoichiometricNet(input_size=14, hidden_size=128, output_size=6)
    state_dict = torch.load(MODEL_PATH,
                            map_location=CPU_DEVICE,
                            weights_only=True)
    model.load_state_dict(state_dict)

    mean_mape, high_error_count, low_error_count = evaluate_batches(
        model, evaluation_loader, mape_threshold=0.1)

    print(f"Mean Validation MAPE:     {mean_mape:.4f}")
    print(f"Scores above 0.1 (Error): {high_error_count}")
    print(f"Scores below 0.1 (Good):  {low_error_count}")

    evaluation_data_actual = BasicDataset(
        str(EVALUATION_INPUT_PATH),
        str(ACTUAL_OUTPUT_PATH),
    )

    delta_t_mape, mass_frac_mape, coverages_mape = evaluate_physical_states(
        model, evaluation_data, evaluation_data_actual, MECHANISM_PATH)

    mape_buckets(delta_t_mape, mass_frac_mape, coverages_mape)


if __name__ == "__main__":
    main()
