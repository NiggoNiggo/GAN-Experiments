from .trainer import GANTrainer
import torch
from testing.create_image import SampleImages

class VanillaGANTrainer(GANTrainer):
    def __init__(self, gen, disc, data_loader, loss_fn, optim_gen_strat, optim_disc_strat, latent_dim,device="cuda"):
        super().__init__(gen, disc, data_loader, loss_fn, optim_gen_strat, optim_disc_strat,latent_dim)
        self.latent_dim = latent_dim
        self.device = device
        self.verbose = False #default False
        
        self.gen.to(device)
        self.disc.to(device)
        # optimizers should reference parameters on the correct device; rebuild them now
        self.optim_gen = optim_gen_strat.build_optim(self.gen)
        self.optim_disc = optim_disc_strat.build_optim(self.disc)

        # create the image plotter now that generator has been moved to the correct device
        self.image_plotter = SampleImages(self.gen, self.latent_dim)

    def train_disc(self, real_data):
        batch_size, real, labels = self.ensure_correct_input(real_data)
        
        # flatten each example but keep batch dimension
        real_flat = real.view(batch_size, -1)
        real_pred = self.disc(real_flat, self.verbose)
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_data = self.gen(z).detach()
        # shape should already be (batch, output_dim)
        fake_pred = self.disc(fake_data, self.verbose)
        d_loss = self.loss_fn.disc_loss(real_pred, fake_pred)
        

        self.optim_disc.zero_grad()
        d_loss.backward()
        self.optim_disc.step()

        return d_loss.item()

    def train_gen(self, batch_size):
        real_labels = torch.ones(batch_size, 1, device=self.device)

        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_data = self.gen(z)
        fake_flat = fake_data.view(batch_size, -1)
        fake_pred = self.disc(fake_flat, self.verbose)
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