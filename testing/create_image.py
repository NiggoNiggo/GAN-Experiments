import torch
from abc import ABC, abstractmethod
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

class SampleImage(ABC):
    def __init__(self,
                 gen,
                 latent_dim:int):
        self.gen = gen
        self.latent_dim = latent_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sample_images(self):
        raise NotImplementedError()
    
    
class SampleNormalImages(SampleImage):
    def __init__(self, gen, latent_dim):
        super().__init__(gen=gen, latent_dim=latent_dim)
        self.gen = gen
        self.latent_dim = latent_dim
        

    def sample_images(self, num_img: int):
        all_noise = torch.randn(num_img, self.latent_dim,device=self.device)
        with torch.no_grad():
            self.gen.eval()
            fake_data = self.gen(all_noise).detach()
            self.gen.train()
        return fake_data  # shape: (num_img, C, H, W) oder (num_img, H*W) bei linear

    def plot_images_grid(self, num_img: int, nrow: int = 4, normalize: bool = True):
        """
        Plots a grid of generated images using torchvision.utils.make_grid.
        - num_img: number of images to generate
        - nrow: number of images per row
        - normalize: scale images to [0,1] for plotting
        """
        images = self.sample_images(num_img)  # Tensor: (N, C, H, W)
        
        # Flattened output von Linear GAN zu 2D konvertieren
        if images.dim() == 2:  # shape: (N, H*W)
            side = int(images.shape[1] ** 0.5)
            images = images.view(-1, 1, side, side)
        
        grid = make_grid(images, nrow=nrow, normalize=normalize)
        np_grid = grid.permute(1, 2, 0).cpu().numpy()

        plt.figure(figsize=(nrow * 2, (num_img // nrow + 1) * 2))
        plt.imshow(np_grid.squeeze(), cmap='gray' if np_grid.shape[2] == 1 else None)
        plt.axis('off')
        plt.show()