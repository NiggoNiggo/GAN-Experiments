from datasets import load_dataset
from torch.utils.data import Dataset
from core.registries import DATASETS

@DATASETS.registry("celebA")
class CelebAHQ(Dataset):
    def __init__(self,
                 data_path,
                 transform,
                 return_labels:False):
        super().__init__()
        self.dataset = load_dataset("korexyz/celeba-hq-256x256",split="train")
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,idx):
        return self.transform(self.dataset[idx]["image"])