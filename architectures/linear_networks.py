import torch.nn as nn
from .layers import LinearLayer
import torch

class LinearGenerator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearGenerator, self).__init__()
        self.model = nn.Sequential(
            LinearLayer(input_dim, 128),
            LinearLayer(128, 256),
            LinearLayer(256, 512),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )
    def forward(self, x):
        return self.model(x)


class LinearDiscriminator(nn.Module)        :
    def __init__(self, input_dim):
        super(LinearDiscriminator, self).__init__()
        self.model = nn.Sequential(
            LinearLayer(input_dim, 512),
            LinearLayer(512, 256),
            LinearLayer(256, 128),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    input_size = 32
    latent_dim = 100
    noise = torch.randn((1,latent_dim))
    image = torch.randn((1,input_size,input_size))
    gen = LinearGenerator(latent_dim, input_size*input_size)
    disc = LinearDiscriminator(input_size*input_size)
    fake = gen(noise).view(1,input_size,input_size)
    pred = disc(fake.flatten())
    print(image.shape,fake.shape,pred.shape)