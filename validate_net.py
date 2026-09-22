import torch
import torch.nn as nn

from src.net import Net, mape
from src.dataset_reactor import ReactorDataset

MODEL_PATH = ""
INPUT_DATA_PATH = ""
OUTPUT_DATA_PATH = ""

device = torch.device('cpu')
model = Net()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))

dataset = ReactorDataset(INPUT_DATA_PATH, OUTPUT_DATA_PATH)

for i in range(len(dataset)):
    x = dataset.__getitem__(i)[0].unsqueeze(0).to(device)
    y = dataset.__getitem__(i)[1].unsqueeze(0).to(device)

    prediction = model.predict(x)

    mape_error = mape(prediction, y)

    print(f"Sample {i}: MAPE Error: {mape_error}")







