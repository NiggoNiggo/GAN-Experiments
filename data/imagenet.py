from torch.utils.data import Dataset
from torchvision.datasets import ImageNet as ds_image

from core.registries import DATASETS


@DATASETS.registry("ImageNet")
class ImageNet(Dataset):
    def __init__(self, root: str, transform, return_labels: bool = False):
        self.dataset = ds_image(root, split="train")
        self.transform = transform
        self.return_labels = return_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        data = self.transform(data)

        if self.return_labels:
            return data, label
        return data