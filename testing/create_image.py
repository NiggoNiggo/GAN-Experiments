import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


class SampleImages:
    def __init__(self, gen, latent_dim: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gen = gen.to(self.device)
        self.latent_dim = latent_dim

    # -------------------------
    # Noise automatisch korrekt erzeugen
    # -------------------------
    def _sample_noise(self, num_img: int):
        try:
            # Testweise DCGAN-Shape probieren
            z = torch.randn(num_img, self.latent_dim, 1, 1, device=self.device)
            with torch.no_grad():
                _ = self.gen(z)
            return z
        except:
            # Fallback auf MLP
            return torch.randn(num_img, self.latent_dim, device=self.device)

    # -------------------------
    # Images generieren
    # -------------------------
    def sample_images(self, num_img: int):
        z = self._sample_noise(num_img)

        self.gen.eval()
        with torch.no_grad():
            fake_data = self.gen(z)
        self.gen.train()

        return fake_data

    # -------------------------
    # Plot Grid (MLP + DCGAN kompatibel)
    # -------------------------
    def plot_images_grid(self, num_img: int, nrow: int = 4, normalize: bool = True):

        images = self.sample_images(num_img)

        # Falls MLP flattened Output
        if images.dim() == 2:
            side = int(images.shape[1] ** 0.5)
            images = images.view(-1, 1, side, side)

        grid = make_grid(images, nrow=nrow, normalize=normalize)
        np_grid = grid.permute(1, 2, 0).cpu().numpy()

        plt.figure(figsize=(nrow * 2, (num_img // nrow + 1) * 2))
        plt.imshow(np_grid.squeeze(), cmap='gray' if np_grid.shape[2] == 1 else None)
        plt.axis('off')
        plt.savefig("test.png")
        plt.close()