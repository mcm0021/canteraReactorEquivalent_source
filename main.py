import torch
import numpy as np

from torch.utils.data import DataLoader
from torch.optim import Adam

from src.data_generation import generate_input_output_data
from src.dataset_reactor import ReactorDataset
from src.net import Net, preprocess_data, loss_function

# input_data, input_data_frame, output_data, output_data_frame = generate_input_output_data(10000, 5, 4, 5, [273.15, 1000.0], [1e5, 2e6], [0.2, 0.8], [-6.0, 0.0], [0.0, 4.0])

# np.save('input_data.npy', input_data)
# np.save('output_data.npy', output_data)

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

training_data, validation_data = torch.utils.data.random_split(dataset, [0.8, 0.2])

training_loader = DataLoader(training_data, batch_size=128, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=128, shuffle=False)

model = Net(scalar_input_size=5, species_size=4, coverages_size=5, hidden_size=128, scalar_output_size=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
if device.type == "cuda":
    print("Using GPU for training")

epochs = 5000
optimizer = Adam(model.parameters(), lr=0.001)

for epoch in range(epochs): 
    model.train(True)
    for inputs, targets in training_loader: 
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        predictions = model(inputs)
        tp_targets = targets[:, :2]
        species_targets = targets[:, 2:6]
        coverages_targets = targets[:, 6:]
        loss = loss_function(predictions, (tp_targets, species_targets, coverages_targets))
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
                tp_targets = targets[:, :2]
                species_targets = targets[:, 2:6]
                coverages_targets = targets[:, 6:]
                loss_v = loss_function(predictions, (tp_targets, species_targets, coverages_targets))
                validation_loss += loss_v.item()

        print(f"Epoch {epoch}, Loss: {loss.item()}")
        print(f"Validation Loss: {validation_loss / len(validation_loader)}")
