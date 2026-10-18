import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from torch.optim import Adam

from nets.stoichiometric_net import StoichiometricNet
from datasets.dataset_sotichiometry import StoichiometryDataset
from nets.reactor_net import min_max_scaling

TRAIN_INPUT_DATA_PATH = "data/input_train.npy"
TRAIN_OUTPUT_DATA_PATH = "data/output_train.npy"
VALIDATION_INPUT_DATA_PATH = "data/input_val.npy"
VALIDATION_OUTPUT_DATA_PATH = "data/output_val.npy"

def preprocess_stoichiometry(x):
    min_bounds = torch.tensor([273.15, 1e5, 0.2, 1e-0, 1])
    max_bounds = torch.tensor([1000.0, 2e6, 0.8, 1e-6, 1e4])
    x[:5] = min_max_scaling(x[:5],
                            min_bounds=min_bounds,
                            max_bounds=max_bounds)
    return x


train_data = StoichiometryDataset(
    TRAIN_INPUT_DATA_PATH,
    TRAIN_OUTPUT_DATA_PATH,
    Train=True,
    transform_input=preprocess_stoichiometry,
)

train_scalers = train_data.get_output_scalers()

validation_data = StoichiometryDataset(
    VALIDATION_INPUT_DATA_PATH,
    VALIDATION_OUTPUT_DATA_PATH,
    Train=False,
    transform_input=preprocess_stoichiometry,
    output_scalers=train_scalers)

training_loader = DataLoader(train_data, batch_size=128, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=128, shuffle=False)

model = StoichiometricNet(14, 128, 6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if device.type == "cuda":
    print("Using GPU for training")

epochs = 5000
optimizer = Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(epochs):
    model.train()
    for inputs, targets in training_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

    if epoch % 1000 == 0:
        model.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for batch in validation_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                predictions = model(inputs)
                loss_v = loss_fn(predictions, targets)
                validation_loss += loss_v.item()

        print(f"Epoch {epoch}, Loss: {loss.item()}")
        print(f"Validation Loss: {validation_loss / len(validation_loader)}")

torch.save(model.state_dict(), "model.pth")
