from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import torch
from sklearn.metrics import mean_absolute_percentage_error
from torch.utils.data import DataLoader

from datasets.dataset_reactor import ReactorDataset
from datasets.dataset_sotichiometry import StoichiometryDataset
from nets.stoichiometric_net import StoichiometricNet
from src.reactor import get_reactor_constants, initialize_reactor, initialize_variables
from src.stoichiometry import (
    build_stoichiometric_matrix,
    compute_stoichiometric_basis,
    state_to_amounts,
)
from utils.preprocessing import preprocess_stoichiometry, get_bounds

CPU_DEVICE = torch.device("cpu")

EVALUATION_INPUT_PATH = Path("data/input_val.npy")
EVALUATION_OUTPUT_PATH = Path("data/output_val.npy")
ACTUAL_OUTPUT_PATH = Path("data/output_actual_val.npy")
MODEL_PATH = Path("model.pth")
MECHANISM_PATH = Path(
    __file__).parent / "mechanisms" / "coOxidationRhAlumina.yaml"

INPUT_MIN_BOUNDS, INPUT_MAX_BOUNDS = get_bounds()

OUTPUT_MEAN = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
OUTPUT_STD = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)


def reverse_scaling(data: torch.Tensor) -> torch.Tensor:
    """Reverses the z-score scaling on CPU using pre-allocated mean and std."""
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

        # A non-finite model output must not make the complete metric NaN.
        # Count such entries as a large error instead.
        relative_error = torch.abs(
            (targets - predictions) / torch.clamp(torch.abs(targets), min=eps))
        relative_error = torch.nan_to_num(
            relative_error,
            nan=1.0e6,
            posinf=1.0e6,
            neginf=1.0e6,
        )
        batch_mape = torch.mean(relative_error, dim=1)
        print(f"batch_mape: {batch_mape}")
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
    raw_dataset: ReactorDataset,
    mechanism_path: Path,
    metric_fn: Callable = mean_absolute_percentage_error,
) -> None:
    """
    Evaluates physical state reconstruction (T, mass fractions, coverages) on CPU.
    """
    model.eval()
    gas, solid, surface = initialize_variables(mechanism_path)
    num_gas_species = len(gas.species_names)

    S, _ = build_stoichiometric_matrix(gas, surface)
    Q, _, _ = compute_stoichiometric_basis(S)

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

        print(f"T_pred: {pred_T:.2f} K | "
              f"T Abs Err: {t_abs_err:.2f} K (MAPE: {t_mape:.4f}) | "
              f"Mass Frac MAPE: {mass_frac_mape:.4f} | "
              f"Coverages MAPE: {coverages_mape:.4f}")


def main() -> None:
    evaluation_data = StoichiometryDataset(
        str(EVALUATION_INPUT_PATH),
        str(EVALUATION_OUTPUT_PATH),
        Train=False,
        transform_input=preprocess_stoichiometry,
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

    # evaluation_data_actual = ReactorDataset(
    #     str(EVALUATION_INPUT_PATH),
    #     str(ACTUAL_OUTPUT_PATH),
    #     transform=None,
    # )
    # evaluate_physical_states(model, evaluation_data, evaluation_data_actual,
    #                          MECHANISM_PATH)


if __name__ == "__main__":
    main()
