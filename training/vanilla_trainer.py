from .trainer import GANTrainer
import torch

class VanillaGANTrainer(GANTrainer):
    def __init__(self, gen, disc, data_loader, loss_fn, optim_gen, optim_disc, latent_dim, device="cuda"):
        super().__init__(gen, disc, data_loader, loss_fn, optim_gen, optim_disc)
        self.latent_dim = latent_dim
        self.device = device
        
        self.gen.to(device)
        self.disc.to(device)

    def train_disc(self, real_data):
        batch_size = real_data.size(0)
        
        real_data = real_data.to(self.device)
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)

        real_pred = self.disc(real_data)
        real_loss = self.loss_fn(real_pred, real_labels)

        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_data = self.gen(z).detach()  # detach wichtig!
        fake_pred = self.disc(fake_data)
        fake_loss = self.loss_fn(fake_pred, fake_labels)
        d_loss = real_loss + fake_loss

        self.optim_disc.zero_grad()
        d_loss.backward()
        self.optim_disc.step()

        return d_loss.item()

    def train_gen(self, batch_size):
        real_labels = torch.ones(batch_size, 1).to(self.device)

        z = torch.randn(batch_size, self.latent_dim).to(self.device)
        fake_data = self.gen(z)
        pred = self.disc(fake_data)
        g_loss = self.loss_fn(pred, real_labels)
        self.optim_gen.zero_grad()
        g_loss.backward()
        self.optim_gen.step()

        return g_loss.item()


    def train_step(self, batch):
        real_data = batch

        d_loss = self.train_disc(real_data)
        g_loss = self.train_gen(real_data.size(0))

        return d_loss, g_loss