from torch.utils.data import Dataset

class ReactorDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.data = list(zip(input_data, output_data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]