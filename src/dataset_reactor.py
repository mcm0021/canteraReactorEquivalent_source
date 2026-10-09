import numpy as np
import torch

from torch.utils.data import Dataset
from src.net import preprocess_data

class ReactorDataset(Dataset):
    def __init__(self, input_data_path, output_data_path, transform=None):
        self.transform = transform

        with open(input_data_path, 'rb') as f:
            input_data = np.load(f)

        with open(output_data_path, 'rb') as f:
            output_data = np.load(f)

        self.input_length = input_data.shape[1]

        input_data, output_data = preprocess_data(
            input_data,
            output_data, 
            scalar_input_max_bounds=np.array([1000.0, 2e6, 0.8, 1e-6, 1e4]),
            scalar_input_min_bounds=np.array([273.15, 1e5, 0.2, 1e-0, 1]),
            scalar_output_max_bounds=np.array([1000.0, 2e6]),
            scalar_output_min_bounds=np.array([273.15, 1e5]),
            )
        
        self.data = np.concatenate((input_data, output_data), axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input = torch.tensor(self.data[idx, :self.input_length], dtype=torch.float32)
        target = torch.tensor(self.data[idx, self.input_length:], dtype=torch.float32)

        if self.transform:
            input = self.transform(input)
            target = self.transform(target)

        return input, target