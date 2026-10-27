import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def get_ranges():
    TEMPERATURE_RANGE = [300.0, 1000.0]
    PRESSURE_RANGE = [1e5, 2e6]
    POROSITY_RANGE = [0.2, 0.8]
    ADVANCE_TIME_RANGE = [-6.0, 0.0]  # Note: Expects log10(dt) in [-6, 0]!
    AREAS_RANGE = [0.0, 4.0]  # Note: Expects log10(area) in [0, 4]!
    return (
        TEMPERATURE_RANGE,
        PRESSURE_RANGE,
        POROSITY_RANGE,
        ADVANCE_TIME_RANGE,
        AREAS_RANGE,
    )


def get_bounds(device=None, dtype=torch.float32):
    ranges = torch.tensor(get_ranges(), dtype=dtype, device=device)
    min_bounds = ranges[:, 0]
    max_bounds = ranges[:, 1]
    return min_bounds, max_bounds


def min_max_scaling(data: torch.Tensor, min_bounds: torch.Tensor,
                    max_bounds: torch.Tensor):
    return (data - min_bounds) / (max_bounds - min_bounds)


def preprocess_stoichiometry(x: torch.Tensor):
    # Ensure x is a tensor and check for inf/nan before processing
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    return preprocess_inputs(x[0], x[1], x[2], x[3], x[4], x[5:9], x[9:])


def preprocess_inputs(
    temp_data: torch.Tensor,
    pressure_data: torch.Tensor,
    porosity_data: torch.Tensor,
    advance_time_data: torch.Tensor,
    catalytic_surface_data: torch.Tensor,
    mass_fractions: torch.Tensor,
    surface_coverages: torch.Tensor,
):

    # Log transform for values that span several orders of magnitude
    advance_time_data = torch.log10(advance_time_data)
    catalytic_surface_data = torch.log10(catalytic_surface_data)

    # Min Max scaling for scalar inputs (temperature, pressure, porosity, advance_time, catalytic_surface)
    scalar_input = torch.cat((
        temp_data.unsqueeze(-1),
        pressure_data.unsqueeze(-1),
        porosity_data.unsqueeze(-1),
        advance_time_data.unsqueeze(-1),
        catalytic_surface_data.unsqueeze(-1),
    ),
                             dim=-1)

    min_bounds, max_bounds = get_bounds(device=scalar_input.device,
                                        dtype=scalar_input.dtype)

    scalar_input = min_max_scaling(scalar_input,
                                   min_bounds=min_bounds,
                                   max_bounds=max_bounds)

    # Safe square-root transform in pure PyTorch (clamping prevents NaN from -1e-16 noise)
    # mass_fractions = torch.sqrt(torch.clamp(mass_fractions, min=0.0))
    # surface_coverages = torch.sqrt(torch.clamp(surface_coverages, min=0.0))

    return torch.cat((scalar_input, mass_fractions, surface_coverages))


def fit_preprocess_outputs(y: np.ndarray | torch.Tensor):
    """Fits the StandardScaler on training targets y (shape: [N, num_outputs])."""
    y_np = y.detach().cpu().numpy() if isinstance(
        y, torch.Tensor) else np.asarray(y)

    scaler = StandardScaler()
    scaler.fit(y_np)

    # Protect against constant columns (std == 0) in small 100-sample datasets
    scaler.scale_ = np.where(scaler.scale_ < 1e-12, 1.0, scaler.scale_)

    y_scaled = torch.as_tensor(scaler.transform(y_np), dtype=torch.float32)
    return y_scaled, scaler


def transform_outputs(
    stoich_coordinates: torch.Tensor,
    delta_T: torch.Tensor,
    scaler: StandardScaler,
):
    """
    Transforms (stoich_coordinates, delta_T) using a fitted StandardScaler.
    Matches predict_real order: stoich_coordinates FIRST, delta_T LAST!
    """
    if delta_T.ndim == 1:
        delta_T = delta_T.unsqueeze(-1)
    if stoich_coordinates.ndim == 1:
        stoich_coordinates = stoich_coordinates.unsqueeze(0)

    # Concatenate along feature dimension (dim=-1): [z_pred, delta_T]
    combined = torch.cat((stoich_coordinates, delta_T), dim=-1)
    combined_np = combined.detach().cpu().numpy()

    scaled_np = scaler.transform(combined_np)
    return torch.as_tensor(scaled_np,
                           dtype=stoich_coordinates.dtype,
                           device=stoich_coordinates.device)
