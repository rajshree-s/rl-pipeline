import json

from torch.utils.data import Dataset


class QuestionDataset(Dataset):
    def __init__(self, data):
        if isinstance(data, str):
            with open(data, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
