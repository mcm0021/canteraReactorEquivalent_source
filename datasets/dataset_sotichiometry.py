import numpy as np
import torch

from torch.utils.data import Dataset
from nets.reactor_net import min_max_scaling 

class StoichiometryDataset(Dataset):
    def __init__(self, input_data_path, output_data_path, Train=False, transform_input=None, transform_output=None, output_scalers=None):
        self.Train = Train
        self.transform_input = transform_input
        self.transform_output = transform_output

        with open(input_data_path, 'rb') as f:
            self.input_data = np.load(f)

        with open(output_data_path, 'rb') as f:
            self.output_data = np.load(f)

        if self.Train:
            self.min_scalers_output = torch.tensor(np.min(self.output_data, axis=0), dtype=torch.float32)
            self.max_scalers_output = torch.tensor(np.max(self.output_data, axis=0), dtype=torch.float32)
        else:
            self.min_scalers_output, self.max_scalers_output = output_scalers


    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input = torch.tensor(self.input_data[idx], dtype=torch.float32)
        target = torch.tensor(self.output_data[idx], dtype=torch.float32)

        if self.transform_input:
            input = self.transform_input(input)

        target = min_max_scaling(target, self.min_scalers_output, self.max_scalers_output)

        return input, target
    
    def get_output_scalers(self):
        return self.min_scalers_output, self.max_scalers_output