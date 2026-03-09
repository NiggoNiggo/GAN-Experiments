import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from abc import ABC, abstractmethod


class AbstractSampleImages(ABC):

    def __init__(self, gen, latent_dim: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gen = gen.to(self.device)
        self.latent_dim = latent_dim

    @abstractmethod
    def sample_images(self, num_img: int):
        pass

    def plot_images_grid(self, images, num_img: int, nrow: int = 4, normalize: bool = True):

        grid = make_grid(images, nrow=nrow, normalize=normalize)
        np_grid = grid.permute(1, 2, 0).cpu().numpy()

        plt.figure(figsize=(nrow * 2, (num_img // nrow + 1) * 2))
        plt.imshow(np_grid.squeeze(), cmap='gray' if np_grid.shape[2] == 1 else None)
        plt.axis('off')
        plt.savefig("test.png")
        plt.close()
    

class LinearGANImageSampler(AbstractSampleImages):

    def sample_images(self, num_img: int):

        z = torch.randn(num_img, self.latent_dim, device=self.device)

        self.gen.eval()
        with torch.no_grad():
            fake_data = self.gen(z)
        self.gen.train()

        # reshape flattened output
        side = int(fake_data.shape[1] ** 0.5)
        fake_data = fake_data.view(-1, 1, side, side)

        return fake_data

class ConvGANImageSampler(AbstractSampleImages):

    def sample_images(self, num_img: int):

        z = torch.randn(num_img, self.latent_dim, 1, 1, device=self.device)

        self.gen.eval()
        with torch.no_grad():
            fake_data = self.gen(z)
        self.gen.train()

        return fake_data
    

class ConditionalGANImageSampler(AbstractSampleImages):

    def __init__(self, gen, latent_dim: int, num_classes: int):
        super().__init__(gen, latent_dim)
        self.num_classes = num_classes

    def sample_images(self, num_img: int):

        z = torch.randn(num_img, self.latent_dim, 1, 1, device=self.device)

        labels = torch.arange(self.num_classes,device=self.device).repeat(num_img // self.num_classes)

        self.gen.eval()
        with torch.no_grad():
            fake_data = self.gen(z, labels)
        self.gen.train()

        return fake_data