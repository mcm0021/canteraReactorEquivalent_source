import numpy as np

from src.data_generation_stoichiometric import run_reactor
from src.data_generation import generate_input_output_data
from utils.preprocessing import get_ranges

NUM_SAMPLES = 1000
INPUT_DATA_PATH = "data/input_data.npy"
OUTPUT_DATA_PATH = "data/output_data_stoichiometric.npy"

temperature_range, pressure_range, porosity_range, advance_time_range, areas_range = get_ranges(
)

input_data, input_data_frame, actual_output_data, actual_output_data_frame = generate_input_output_data(
    NUM_SAMPLES, 5, 4, 5, temperature_range, pressure_range, porosity_range,
    advance_time_range, areas_range)

output_data, output_data_frame = run_reactor(input_data)

rng = np.random.default_rng(seed=42)
indices = rng.permutation(len(input_data))
split = int(0.8 * len(indices))

train_idx = indices[:split]
val_idx = indices[split:]

input_train = input_data[train_idx]
input_val = input_data[val_idx]
output_train = output_data[train_idx]
output_val = output_data[val_idx]

np.save(INPUT_DATA_PATH, input_data)
np.save(OUTPUT_DATA_PATH, output_data)
np.save("data/output_data_frame.npy", output_data_frame.to_numpy())
np.save("data/input_train.npy", input_train)
np.save("data/input_val.npy", input_val)
np.save("data/output_train.npy", output_train)
np.save("data/output_val.npy", output_val)
np.save("data/output_actual_val.npy", actual_output_data)
