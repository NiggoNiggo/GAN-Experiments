
import torchvision.transforms as transforms
import torch
import os
from pathlib import Path
from PIL import Image

class CelebADataset(torch.utils.data.Dataset):
    def __init__(self,
                 path:str,
                 transforms):
        self.transforms = transforms
        self.path = path
        self.data = []
        for r,d,f in os.walk(Path(self.path)):
            self.data.extend([Path(r) / file for file in f])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        item = Image.open(self.data[idx]).convert("RGB")
        return self.transforms(item)




if __name__ == "__main__":
    transform = transforms.Compose([
    transforms.Resize((16, 16)),      
    transforms.ToTensor(),             
    transforms.Normalize((0.5,), (0.5,), (0.5,))  
    ])


    path = r"D:\Datasets\celebA\img_align_celeba"
    data = CelebADataset(path,transform)
    print(len(data))
    x = next(iter(data))
    print(x.shape)