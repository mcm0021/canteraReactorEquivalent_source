import torch
import torch.nn as nn
import numpy as np

def preprocess_data(input_data, output_data, scalar_input_size=5, scalar_output_size=2, MinMax=True):
    """
    Preprocess the input data for the neural network.
    Returns:
    - A PyTorch tensor containing the preprocessed data.
    """

    if MinMax: 
        # Min max scaling
        input_data[:, :scalar_input_size] = (input_data[:, :scalar_input_size] - np.min(input_data[:, :scalar_input_size], axis=0)) / (np.max(input_data[:, :scalar_input_size], axis=0) - np.min(input_data[:, :scalar_input_size], axis=0))
        output_data[:, :scalar_output_size] = (output_data[:, :scalar_output_size] - np.min(output_data[:, :scalar_output_size], axis=0)) / (np.max(output_data[:, :scalar_output_size], axis=0) - np.min(output_data[:, :scalar_output_size], axis=0))
    else:
        # Standardization (Z-score normalization)
        input_data[:, :scalar_input_size] = (input_data[:, :scalar_input_size] - np.mean(input_data[:, :scalar_input_size], axis=0)) / np.std(input_data[:, :scalar_input_size], axis=0)
        output_data[:, :scalar_output_size] = (output_data[:, :scalar_output_size] - np.mean(output_data[:, :scalar_output_size], axis=0)) / np.std(output_data[:, :scalar_output_size], axis=0)
        
        
    return torch.tensor(input_data, dtype=torch.float32), torch.tensor(output_data, dtype=torch.float32)

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
    tp_predictions, species_predictions, coverages_predictions = predictions
    tp_targets, species_targets, coverages_targets = targets

    np.abs((tp_predictions - tp_targets) / tp_targets).mean() * 100, \
    np.abs((species_predictions - species_targets) / species_targets).mean() * 100, \
    np.abs((coverages_predictions - coverages_targets) / coverages_targets).mean() * 100

class Net(nn.Module):
    def __init__(self, scalar_input_size, species_size, coverages_size, hidden_size, scalar_output_size):
        super(Net, self).__init__()

        self.shared_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(scalar_input_size + species_size + coverages_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.tp_head = nn.Linear(hidden_size, scalar_output_size)
        self.species_head = nn.Sequential(
            nn.Linear(hidden_size, species_size),
            nn.Softmax(dim=1)
        )
        self.coverages_head = nn.Sequential(
            nn.Linear(hidden_size, coverages_size),
            nn.Softmax(dim=1)
        )

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
