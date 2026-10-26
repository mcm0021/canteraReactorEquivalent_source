import numpy as np
import torch

from sklearn.preprocessing import FunctionTransformer, StandardScaler


def get_ranges():
    TEMPERATURE_RANGE = [300, 1000.0]
    PRESSURE_RANGE = [1e5, 2e6]
    POROSITY_RANGE = [0.2, 0.8]
    ADVANCE_TIME_RANGE = [-6.0, 0.0]
    AREAS_RANGE = [0.0, 4.0]
    return TEMPERATURE_RANGE, PRESSURE_RANGE, POROSITY_RANGE, ADVANCE_TIME_RANGE, AREAS_RANGE


def get_bounds():
    temp_range, press_range, porosity_range, advance_time_range, areas_range = get_ranges(
    )
    return np.min(np.array([
        temp_range, press_range, porosity_range, advance_time_range,
        areas_range
    ]),
                  axis=1), np.max(np.array([
                      temp_range, press_range, porosity_range,
                      advance_time_range, areas_range
                  ]),
                                  axis=1)


def preprocess_stoichiometry(x):
    x = preprocess_inputs(x[:5], x[5:9], x[9:14])
    # y, _ = preprocess_outputs(y[:1], y[1:])
    # return x, y
    return x


def preprocess_inputs(scalar_input, mass_fractions, surface_coverages):
    min_bounds, max_bounds = get_bounds()
    # Min-max scaling for scalar inputs
    scalar_input = min_max_scaling(scalar_input,
                                   min_bounds=min_bounds,
                                   max_bounds=max_bounds)
    # Transform mass fractions and surface coverages using a square root function
    # Values are transformed to reduce skewness and improve model performance (must be positive)
    # mass_fractions = transform(mass_fractions, np.sqrt)
    # surface_coverages = transform(surface_coverages, np.sqrt)
    return torch.concatenate((scalar_input, mass_fractions, surface_coverages))


def preprocess_outputs(delta_T, stoich_coordinates, scaler: StandardScaler):
    return np.concatenate((delta_T, stoich_coordinates)), scaler.transform(
        np.concatenate((delta_T, stoich_coordinates)))


def min_max_scaling(data, min_bounds, max_bounds):
    return (data - min_bounds) / (max_bounds - min_bounds)


def transform(data, func):
    transform = FunctionTransformer(func=func, validate=True)
    return transform.fit_transform(data)
