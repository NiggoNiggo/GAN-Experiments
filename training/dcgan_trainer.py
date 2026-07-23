import os
from .trainer import GANTrainer
import torch
from testing.create_image import ConvGANImageSampler
from core.registries import TRAINERS

@TRAINERS.registry("dcgan")
class DCGANTrainer(GANTrainer):
    def __init__(self, 
                cfg):
        super().__init__(cfg)
        
        self.init_models()

    def train_disc(self, real_data):
        batch_size, real, labels = self.ensure_correct_input(real_data)
        
        real_pred = self.disc(real).view(-1)
        z = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
        fake_data = self.gen(z).detach()
        # shape should already be (batch, output_dim)
        fake_pred = self.disc(fake_data).view(-1)
        d_loss = self.loss_fn.disc_loss(real_pred, fake_pred)
        
        self.optim_disc.zero_grad()
        d_loss.backward()
        self.optim_disc.step()

        return d_loss.item()

    def train_gen(self, batch_size):
        z = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
        fake_data = self.gen(z)
        fake_pred = self.disc(fake_data).view(-1)
        g_loss = self.loss_fn.gen_loss(fake_pred)
        self.optim_gen.zero_grad()
        g_loss.backward()
        self.optim_gen.step()

        return g_loss.item()


    def train_step(self, batch):
        d_loss = self.train_disc(batch)
        batch_size, real, labels = self.ensure_correct_input(batch)
        g_loss = self.train_gen(batch_size)


        return d_loss, g_loss
    
    