import torch

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from abc import ABC, abstractmethod


class AbstractSampleImages(ABC):

    def __init__(self, gen, latent_dim: int, num_fixed_images: int = 16, seed: int = 42):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.gen = gen.to(self.device)
        self.latent_dim = latent_dim

        # Fixed latent vectors for reproducible visual comparison
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        self.fixed_z = torch.randn(
            num_fixed_images,
            latent_dim,
            1,
            1,
            generator=generator,
            device=self.device
        )

    @abstractmethod
    def sample_images(self, num_img: int):
        pass

    def plot_images_grid(
        self,
        images,
        num_img: int,
        nrow: int = 4,
        filename: str = "sample_grid.png"
    ):
        grid = make_grid(
            images,
            nrow=nrow,
            normalize=True,
            value_range=(-1, 1)
        )

        np_grid = grid.permute(1, 2, 0).cpu().numpy()

        plt.figure(
            figsize=(nrow * 2, (num_img // nrow + 1) * 2)
        )

        plt.imshow(
            np_grid.squeeze(),
            cmap="gray" if np_grid.shape[2] == 1 else None
        )

        plt.axis("off")
        plt.savefig(filename, bbox_inches="tight")
        plt.close()


class ConvGANImageSampler(AbstractSampleImages):

    def sample_images(self, num_img: int):
        z = self.fixed_z[:num_img]
        self.gen.eval()
        with torch.inference_mode():
            fake_data = self.gen(z)
        self.gen.train()
        return fake_data


class ConditionalGANImageSampler(AbstractSampleImages):
    def __init__(
        self,
        gen,
        latent_dim: int,
        num_classes: int,
        num_fixed_images: int = 16,
        seed: int = 42
    ):
        super().__init__(
            gen,
            latent_dim,
            num_fixed_images,
            seed
        )

        self.num_classes = num_classes
        # Fixed labels
        self.fixed_labels = (
            torch.arange(num_classes, device=self.device).repeat((num_fixed_images + num_classes - 1) // num_classes)[:num_fixed_images]
        )

    def sample_images(self, num_img: int):
        z = self.fixed_z[:num_img]
        labels = self.fixed_labels[:num_img]
        self.gen.eval()
        with torch.inference_mode():
            fake_data = self.gen(z, labels)
        self.gen.train()
        return fake_data
