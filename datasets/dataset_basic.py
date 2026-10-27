import numpy as np
import torch

from torch.utils.data import Dataset


class BasicDataset(Dataset):

    def __init__(self,
                 input_data_path,
                 output_data_path,
                 input_transform=None,
                 output_transform=None):
        self.input_transform = input_transform
        self.output_transform = output_transform

        with open(input_data_path, 'rb') as f:
            input_data = np.load(f)

        with open(output_data_path, 'rb') as f:
            output_data = np.load(f)

        self.input_data = input_data
        self.output_data = output_data

    def __len__(self):
        return len(self.input_data), len(self.output_data)

    def __getitem__(self, idx):
        input = self.input_data[idx]
        target = self.output_data[idx]

        if self.input_transform:
            input = self.input_transform(input)
        if self.output_transform:
            target = self.output_transform(target)

        return torch.tensor(input, dtype=torch.float32), torch.tensor(
            target, dtype=torch.float32)
