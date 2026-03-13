from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms as transforms


class CycleDataset(Dataset):

    def __init__(self, path_A, path_B, transform=None):

        self.path_A = path_A
        self.path_B = path_B

        self.files_A = os.listdir(path_A)
        self.files_B = os.listdir(path_B)

        #change this into more flexible for increasing the resolution of cyclegan
        self.transform = transforms.Compose([
            transforms.Resize(286),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, idx):

        img_A = Image.open(
            os.path.join(self.path_A, self.files_A[idx % len(self.files_A)])
        ).convert("RGB")

        img_B = Image.open(
            os.path.join(self.path_B, self.files_B[idx % len(self.files_B)])
        ).convert("RGB")

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}