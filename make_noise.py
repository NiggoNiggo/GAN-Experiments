import torch

num_images = 16
latent_dim = 128

torch.manual_seed(42)

fixed_z = torch.randn(
    num_images,
    latent_dim,
    1,
    1
)

torch.save(fixed_z, "fixed_noise.pt")