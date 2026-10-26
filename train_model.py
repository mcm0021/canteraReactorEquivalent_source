import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import Adam

from nets.stoichiometric_net import StoichiometricNet
from datasets.dataset_sotichiometry import StoichiometryDataset
from utils.preprocessing import preprocess_stoichiometry

TRAIN_INPUT_DATA_PATH = "data/input_train.npy"
TRAIN_OUTPUT_DATA_PATH = "data/output_train.npy"
VALIDATION_INPUT_DATA_PATH = "data/input_val.npy"
VALIDATION_OUTPUT_DATA_PATH = "data/output_val.npy"

train_data = StoichiometryDataset(
    TRAIN_INPUT_DATA_PATH,
    TRAIN_OUTPUT_DATA_PATH,
    Train=True,
    transform_input=preprocess_stoichiometry,
)

train_scalers = train_data.get_output_scalers()
mean, std = train_data.get_output_mean_std()
print("Output mean:", mean)
print("Output std:", std)

validation_data = StoichiometryDataset(
    VALIDATION_INPUT_DATA_PATH,
    VALIDATION_OUTPUT_DATA_PATH,
    Train=False,
    transform_input=preprocess_stoichiometry,
    output_scalers=train_scalers,
    mean=mean,
    std=std)

training_loader = DataLoader(train_data, batch_size=128, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=128, shuffle=False)

model = StoichiometricNet(14, 128, 6)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if device.type == "cuda":
    print("Using GPU for training")

epochs = 5000
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
loss_fn = nn.SmoothL1Loss()

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    num_samples = 0

    for inputs, targets in training_loader:
        inputs = inputs.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.float32)
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        train_loss += loss.item() * batch_size
        num_samples += batch_size

    if epoch % 100 == 0:
        model.eval()
        validation_loss = 0.0
        val_num_samples = 0
        with torch.no_grad():
            for batch in validation_loader:
                inputs, targets = batch
                inputs = inputs.to(device=device, dtype=torch.float32)
                targets = targets.to(device=device, dtype=torch.float32)
                predictions = model(inputs)
                loss_v = loss_fn(predictions, targets)
                batch_size = inputs.size(0)
                validation_loss += loss_v.item() * batch_size
                val_num_samples += batch_size

        epoch_train_loss = train_loss / num_samples
        epoch_validation_loss = validation_loss / val_num_samples
        print(f"Epoch {epoch}, Loss: {epoch_train_loss}")
        print(f"Validation Loss: {epoch_validation_loss}")

torch.save(model.state_dict(), "model.pth")
