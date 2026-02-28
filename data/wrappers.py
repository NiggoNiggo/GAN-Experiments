import torch

class DataWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, has_labels=True):
        self.dataset = dataset
        self.has_labels = has_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if self.has_labels:
            x, y = sample
            return {"x": x, "y": y}
        else:
            return {"x": sample}