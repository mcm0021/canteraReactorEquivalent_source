import numpy as np
import torch

from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class StoichiometryDataset(Dataset):

    def __init__(self,
                 input_data_path,
                 output_data_path,
                 Train=False,
                 transform_input=None,
                 transform_output=None,
                 output_scalers=None,
                 mean=None,
                 std=None):
        self.Train = Train
        self.transform_input = transform_input
        self.transform_output = transform_output

        with open(input_data_path, 'rb') as f:
            self.input_data = np.load(f)

        with open(output_data_path, 'rb') as f:
            self.output_data = np.load(f)

        if self.Train:
            scaler = StandardScaler()
            self.output_data = scaler.fit_transform(self.output_data)
            self.mean = scaler.mean_
            self.std = scaler.scale_
            self.min_scalers_output = np.min(self.output_data, axis=0)
            self.max_scalers_output = np.max(self.output_data, axis=0)
        else:
            self.min_scalers_output, self.max_scalers_output = output_scalers
            self.mean = mean
            self.std = std

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input = torch.tensor(self.input_data[idx], dtype=torch.float32)
        target = torch.tensor(self.output_data[idx], dtype=torch.float32)

        if self.transform_input:
            input = self.transform_input(input)

        if not self.Train:
            target = (target - torch.as_tensor(self.mean, dtype=torch.float32)
                      ) / torch.as_tensor(self.std, dtype=torch.float32)

        return input, target

    def get_output_scalers(self):
        return self.min_scalers_output, self.max_scalers_output

    def get_output_mean_std(self):
        return self.mean, self.std
