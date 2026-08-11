import torch

from .trainer import GANTrainer
from core.registries import TRAINERS



@TRAINERS.registry("wgan")
class WGANTrainer(GANTrainer):
    def __init__(self,cfg):
        super().__init__(cfg)
        self.cfg = cfg
        #how often to train the discriminator
        self.init_models()
    
    def train_step(self, batch):
        #initialise disc error 
        d_loss = 0
        #compute disc error num_k times
        for k in range(self.cfg["params"]["n_crit"]):
            #call error computation
            d_loss += self.train_disc(batch)
        #average the loss
        d_loss /= self.cfg["params"]["n_crit"]
        #train gen once
        batch_size, real, labels = self.ensure_correct_input(batch)
        g_loss = self.train_gen(batch_size)
        return d_loss, g_loss
    
    def train_gen(self,batchsize):
        z = torch.randn(size=(batchsize,self.cfg["params"]["latent_dim"],1,1),device=self.device)
        fake = self.gen(z)
        fake_pred = self.disc(fake).view(-1)
        g_loss = self.loss_fn.gen_loss(fake_pred)
        self.optim_gen.zero_grad()
        g_loss.backward()
        self.optim_gen.step()
        #item destroys the torch graph
        return g_loss.item()

    
    def train_disc(self, real_data):
        batch_size, real, labels = self.ensure_correct_input(real_data)
        real_pred = self.disc(real).view(-1)
        z = torch.randn(batch_size, self.cfg["params"]["latent_dim"], 1, 1, device=self.device)
        fake_data = self.gen(z).detach()
        # shape should already be (batch, output_dim)
        fake_pred = self.disc(fake_data).view(-1)
        if self.cfg["mode"] == "gp":
            #define e for interpolation
            e = torch.rand(
                real.size(0),
                *([1] * (real.dim() - 1)),
                device=real.device
            )
            #apply the interpolation
            x_hat = e * real + (1-e)* fake_data
            x_hat.requires_grad_(True)
            #compute x_hat pred 
            x_hat_pred = self.disc(x_hat).view(-1)
            #call the computation for the actual loss
            d_loss = self.loss_fn.disc_loss(real_pred,
                                            fake_pred,
                                            x_hat,
                                            x_hat_pred,
                                            self.cfg["params"]["lambda"])

        else:
            d_loss = self.loss_fn.disc_loss(real_pred, fake_pred)
        
        self.optim_disc.zero_grad()
        d_loss.backward()
        self.optim_disc.step()
        #if weight clipping to it at the end to the in params given c
        if self.cfg["mode"] == "clipping":
            for p in self.disc.parameters():
                p.data.clamp_(-self.cfg["params"]["c"], self.cfg["params"]["c"])
        return d_loss.item()

