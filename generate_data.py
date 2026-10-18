import numpy as np

from src.data_generation_stoichiometric import run_reactor
from src.data_generation import generate_input_output_data

NUM_SAMPLES = 10
INPUT_DATA_PATH = "data/input_data.npy"
OUTPUT_DATA_PATH = "data/output_data_stoichiometric.npy"

input_data, input_data_frame, output_data, output_data_frame = generate_input_output_data(
    NUM_SAMPLES, 5, 4, 5, [300, 1000.0], [1e5, 2e6], [0.2, 0.8], [-6.0, 0.0],
    [0.0, 4.0])

output_data, output_data_frame = run_reactor(input_data)

split_factor = 0.8
input_train = input_data[:(int(NUM_SAMPLES * split_factor))]
input_val = input_data[(int(NUM_SAMPLES * split_factor)):]

output_train = output_data[:(int(NUM_SAMPLES * split_factor))]
output_val = output_data[(int(NUM_SAMPLES * split_factor)):]

np.save(INPUT_DATA_PATH, input_data)
np.save(OUTPUT_DATA_PATH, output_data)
np.save("data/output_data_frame.npy", output_data_frame.to_numpy())
np.save("data/input_train.npy", input_train)
np.save("data/input_val.npy", input_val)
np.save("data/output_train.npy", output_train)
np.save("data/output_val.npy", output_val)
