import torch
import torch.nn as nn
import numpy as np

from sklearn.preprocessing import FunctionTransformer


def preprocess_data(input_data,
                    output_data,
                    scalar_input_max_bounds,
                    scalar_input_min_bounds,
                    scalar_output_max_bounds,
                    scalar_output_min_bounds,
                    func=lambda x: x**0.5,
                    scalar_input_size=5,
                    scalar_output_size=2,
                    MinMax=True):
    """
    Preprocess the input data for the neural network.
    Returns:
    - A PyTorch tensor containing the preprocessed data.
    """

    if MinMax:
        # Min max scaling with provided bounds
        input_data[:, :scalar_input_size] = min_max_scaling(
            input_data[:, :scalar_input_size], scalar_input_min_bounds,
            scalar_input_max_bounds)
        output_data[:, :scalar_output_size] = min_max_scaling(
            output_data[:, :scalar_output_size], scalar_output_min_bounds,
            scalar_output_max_bounds)
    else:
        # Standardization (Z-score normalization)
        input_data[:, :scalar_input_size] = (
            input_data[:, :scalar_input_size] -
            np.mean(input_data[:, :scalar_input_size], axis=0)) / np.std(
                input_data[:, :scalar_input_size], axis=0)
        output_data[:, :scalar_output_size] = (
            output_data[:, :scalar_output_size] -
            np.mean(output_data[:, :scalar_output_size], axis=0)) / np.std(
                output_data[:, :scalar_output_size], axis=0)

    input_data[:, scalar_input_size:] = transform(
        input_data[:, scalar_input_size:], func)
    output_data[:, scalar_output_size:(scalar_output_size + 4)] = transform(
        output_data[:, scalar_output_size:(scalar_output_size + 4)], func)

    return input_data, output_data


def min_max_scaling(data, min_bounds, max_bounds):
    if torch.is_tensor(data):
        data = data.to(dtype=torch.float64)
        min_bounds = torch.as_tensor(
            min_bounds, dtype=data.dtype, device=data.device)
        max_bounds = torch.as_tensor(
            max_bounds, dtype=data.dtype, device=data.device)

        if min_bounds.shape != max_bounds.shape:
            raise ValueError("min_bounds and max_bounds must have the same shape")
        if data.shape[-1] != min_bounds.numel():
            raise ValueError(
                "The number of data features must match the number of bounds")
        if not torch.isfinite(min_bounds).all() or not torch.isfinite(
                max_bounds).all():
            raise ValueError("Scaling bounds must contain only finite values")

        bounds_range = max_bounds - min_bounds
        if torch.any(bounds_range <= 0):
            raise ValueError("Each max bound must be greater than its min bound")

        scaled_data = (data - min_bounds) / bounds_range
        if not torch.isfinite(scaled_data).all():
            raise ValueError("Min-max scaling produced non-finite values")
        return scaled_data

    data = np.asarray(data, dtype=np.float64)
    min_bounds = np.asarray(min_bounds, dtype=np.float64)
    max_bounds = np.asarray(max_bounds, dtype=np.float64)

    if min_bounds.shape != max_bounds.shape:
        raise ValueError("min_bounds and max_bounds must have the same shape")
    if data.shape[-1] != min_bounds.size:
        raise ValueError(
            "The number of data features must match the number of bounds")
    if not np.isfinite(min_bounds).all() or not np.isfinite(max_bounds).all():
        raise ValueError("Scaling bounds must contain only finite values")

    bounds_range = max_bounds - min_bounds
    if np.any(bounds_range <= 0):
        raise ValueError("Each max bound must be greater than its min bound")

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        scaled_data = (data - min_bounds) / bounds_range

    if not np.isfinite(scaled_data).all():
        raise ValueError("Min-max scaling produced non-finite values")
    return scaled_data


def transform(data, func):
    transform = FunctionTransformer(func=func, validate=True)
    return transform.fit_transform(data)


def denormalize_data_Min_Max(data, min_bounds, max_bounds):
    """
    Denormalize the data using the provided min and max bounds.
    """
    return data * (max_bounds - min_bounds) + min_bounds


def loss_function(predictions, targets):
    """
    Custom loss function that combines MSE for temperature and pressure
    with loss for mass fractions and surface coverages.

    Parameters:
    - predictions: Tuple containing (tp_predictions, species_predictions, coverages_predictions)
    - targets: Tuple containing (tp_targets, species_targets, coverages_targets)

    Returns:
    - Combined loss value.
    """
    tp_predictions, species_predictions, coverages_predictions = predictions
    tp_targets, species_targets, coverages_targets = targets

    # Mean Squared Error for temperature and pressure
    mse_loss = nn.MSELoss()(tp_predictions, tp_targets)

    # Mean Squared Error Loss for mass fractions and surface coverages
    ce_loss_species = nn.MSELoss()(species_predictions, species_targets)
    ce_loss_coverages = nn.MSELoss()(coverages_predictions, coverages_targets)

    total_loss = mse_loss + ce_loss_species + ce_loss_coverages
    return total_loss


def mape(predictions, targets):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) for the predictions.
    Parameters:
    - predictions: Tuple containing (tp_predictions, species_predictions, coverages_predictions)
    - targets: Tuple containing (tp_targets, species_targets, coverages_targets)
    Returns:
    - Tuple containing MAPE for temperature and pressure, mass fractions, and surface coverages.
    """
    tp_predictions, species_predictions, coverages_predictions = predictions
    tp_targets, species_targets, coverages_targets = targets

    return (
        np.abs((tp_predictions - tp_targets) / tp_targets).mean() * 100,
        np.abs(
            (species_predictions - species_targets) / species_targets).mean() *
        100,
        np.abs((coverages_predictions - coverages_targets) /
               coverages_targets).mean() * 100,
    )


class Net(nn.Module):

    def __init__(self, scalar_input_size, species_size, coverages_size,
                 hidden_size, scalar_output_size):
        super(Net, self).__init__()

        self.shared_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(scalar_input_size + species_size + coverages_size,
                      hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU())

        self.tp_head = nn.Linear(hidden_size, scalar_output_size)
        self.species_head = nn.Sequential(nn.Linear(hidden_size, species_size),
                                          nn.Softmax(dim=1))
        self.coverages_head = nn.Sequential(
            nn.Linear(hidden_size, coverages_size), nn.Softmax(dim=1))

    def forward(self, x):
        shared_output = self.shared_layers(x)
        tp_output = self.tp_head(shared_output)
        species_output = self.species_head(shared_output)
        coverages_output = self.coverages_head(shared_output)

        return tp_output, species_output, coverages_output

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            tp_output, species_output, coverages_output = self.forward(x)
        return tp_output, species_output, coverages_output
