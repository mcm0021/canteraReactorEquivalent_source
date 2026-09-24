import numpy as np

from src.data_generation import generate_input_output_data

NUM_SAMPLES = 10
INPUT_DATA_PATH = ""
OUTPUT_DATA_PATH = ""

input_data, input_data_frame, output_data, output_data_frame = generate_input_output_data(NUM_SAMPLES, 5, 4, 5, [273.15, 1000.0], [1e5, 2e6], [0.2, 0.8], [-6.0, 0.0], [0.0, 4.0])

np.save(INPUT_DATA_PATH, input_data)    
np.save(OUTPUT_DATA_PATH, output_data)
