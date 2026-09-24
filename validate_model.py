import torch
import numpy as np

from src.net import Net, mape, preprocess_data 
from src.dataset_reactor import ReactorDataset

MODEL_PATH = ""
INPUT_DATA_PATH = ""
OUTPUT_DATA_PATH = ""

device = torch.device('cpu')
model = Net(scalar_input_size=5, species_size=4, coverages_size=5, hidden_size=128, scalar_output_size=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))

with open('input_data.npy', 'rb') as f:
    input_data= np.load(f)

with open('output_data.npy', 'rb') as f:
    output_data= np.load(f)

input_data, output_data = preprocess_data(input_data,
                                            output_data, 
                                            scalar_input_max_bounds=np.array([1000.0, 2e6, 0.8, 1e-6, 1e4]),
                                            scalar_input_min_bounds=np.array([273.15, 1e5, 0.2, 1e-0, 1]),
                                            scalar_output_max_bounds=np.array([1000.0, 2e6]),
                                            scalar_output_min_bounds=np.array([273.15, 1e5]),
                                           )

dataset = ReactorDataset(input_data, output_data)

for i in range(len(dataset)):
    x, y = dataset[i]

    prediction = model.predict(x.unsqueeze(0))

    targets = (y[:2].unsqueeze(0), y[2:6].unsqueeze(0), y[6:].unsqueeze(0))

    tp, species, coverages = prediction
    tp_targets, species_targets, coverages_targets = targets

    diff = (np.abs(tp - tp_targets), np.abs(species - species_targets), np.abs(coverages - coverages_targets))

    mape_error = mape(prediction, targets)

    print(f"Difference: {diff}")
    print(f"Sample {i}: MAPE Error: {mape_error}")








