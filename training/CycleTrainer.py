from .trainer import GANTrainer
import torch


class CycleGANTrainer(GANTrainer):

    def __init__(self,
                 G_AB,
                 G_BA,
                 D_A,
                 D_B,
                 data_loader,
                 loss_fn,
                 optim_strat,
                 latent_dim,
                 save_path,
                 filename):

        self.G_AB = G_AB
        self.G_BA = G_BA
        self.D_A = D_A
        self.D_B = D_B

        super().__init__(
            gen=None,
            disc=None,
            data_loader=data_loader,
            loss_fn=loss_fn,
            optim_gen_strat=optim_strat,
            optim_disc_strat=optim_strat,
            latent_dim=latent_dim,
            save_path=save_path,
            filename=filename
        )
    
    def train_disc(self, real_A, real_B, fake_A, fake_B):

        self.optim_disc.zero_grad()

        loss_D_A_real = self.loss_fn(self.D_A(real_A), torch.ones_like(self.D_A(real_A)))
        loss_D_A_fake = self.loss_fn(self.D_A(fake_A.detach()), torch.zeros_like(self.D_A(fake_A)))

        loss_D_B_real = self.loss_fn(self.D_B(real_B), torch.ones_like(self.D_B(real_B)))
        loss_D_B_fake = self.loss_fn(self.D_B(fake_B.detach()), torch.zeros_like(self.D_B(fake_B)))

        loss_D = (loss_D_A_real + loss_D_A_fake + loss_D_B_real + loss_D_B_fake) * 0.5

        loss_D.backward()
        self.optim_disc.step()

        return loss_D.item()

    def train_gen(self, real_A, real_B):

        self.optim_gen.zero_grad()

        fake_B = self.G_AB(real_A)
        fake_A = self.G_BA(real_B)

        cycle_A = self.G_BA(fake_B)
        cycle_B = self.G_AB(fake_A)

        # GAN Loss
        loss_G_AB = self.loss_fn(self.D_B(fake_B), torch.ones_like(self.D_B(fake_B)))
        loss_G_BA = self.loss_fn(self.D_A(fake_A), torch.ones_like(self.D_A(fake_A)))

        # Cycle Loss
        loss_cycle_A = torch.nn.functional.l1_loss(cycle_A, real_A)
        loss_cycle_B = torch.nn.functional.l1_loss(cycle_B, real_B)

        loss_cycle = loss_cycle_A + loss_cycle_B

        # Identity Loss
        id_A = self.G_BA(real_A)
        id_B = self.G_AB(real_B)

        loss_id = (
            torch.nn.functional.l1_loss(id_A, real_A) +
            torch.nn.functional.l1_loss(id_B, real_B)
        )

        loss_G = loss_G_AB + loss_G_BA + 10 * loss_cycle + 5 * loss_id

        loss_G.backward()
        self.optim_gen.step()

        return loss_G.item()

    def train_step(self, batch):

        real_A = batch["A"].to(self.device)
        real_B = batch["B"].to(self.device)

        fake_B = self.G_AB(real_A)
        fake_A = self.G_BA(real_B)

        d_loss = self.train_disc(real_A, real_B, fake_A, fake_B)

        g_loss = self.train_gen(real_A, real_B)

        return d_loss, g_loss