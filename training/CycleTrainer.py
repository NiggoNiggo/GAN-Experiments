from .trainer import GANTrainer
import torch
import os
from itertools import chain
import matplotlib.pyplot as plt
from architectures.init_weights import weights_init



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
        self.save_path = save_path  # Set save_path BEFORE init_models()
        
        

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
        
        self.init_models()
        
        self.optim_gen = optim_strat.build_optim(self.G_AB,self.G_BA)
        self.optim_disc = optim_strat.build_optim(self.D_A,self.D_B)
        
        
        self.G_AB.to(self.device)
        self.G_BA.to(self.device)
        self.D_A.to(self.device)
        self.D_B.to(self.device)
        
    
    def train_disc(self, real_A, real_B, fake_A, fake_B):

        self.optim_disc.zero_grad()

        # Discriminator predictions
        pred_real_A = self.D_A(real_A)
        pred_fake_A = self.D_A(fake_A.detach())

        pred_real_B = self.D_B(real_B)
        pred_fake_B = self.D_B(fake_B.detach())

        # Loss über deine Lossklasse
        loss_D_A = self.loss_fn.disc_loss(pred_real_A, pred_fake_A)
        loss_D_B = self.loss_fn.disc_loss(pred_real_B, pred_fake_B)

        loss_D = 0.5 * (loss_D_A + loss_D_B)

        loss_D.backward()
        self.optim_disc.step()

        return loss_D.item()

    def train_gen(self, real_A, real_B):

        self.optim_gen.zero_grad()

        fake_B = self.G_AB(real_A)
        fake_A = self.G_BA(real_B)

        cycle_A = self.G_BA(fake_B)
        cycle_B = self.G_AB(fake_A)

        # Discriminator predictions
        pred_fake_B = self.D_B(fake_B)
        pred_fake_A = self.D_A(fake_A)

        # GAN Loss über deine Klasse
        loss_G_AB = self.loss_fn.gen_loss(pred_fake_B)
        loss_G_BA = self.loss_fn.gen_loss(pred_fake_A)

        # Cycle Loss
        loss_cycle_A = torch.nn.functional.l1_loss(cycle_A, real_A)
        loss_cycle_B = torch.nn.functional.l1_loss(cycle_B, real_B)

        loss_cycle = loss_cycle_A + loss_cycle_B

        # Identity Loss
        id_A = self.G_BA(real_A)
        id_B = self.G_AB(real_B)

        loss_id = (
            torch.nn.functional.l1_loss(id_A, real_A)
            + torch.nn.functional.l1_loss(id_B, real_B)
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

    # def init_models(self):
    #     models_dir = os.path.join(self.save_path, "models")
        
    #     if os.path.exists(models_dir):
    #         # Try to find and load the latest trained models
    #         try:
    #             # Find latest epoch
    #             files = os.listdir(models_dir)
    #             epochs = []
    #             for f in files:
    #                 if f.startswith("G_AB_epoch_"):
    #                     epoch_num = int(f.replace("G_AB_epoch_", "").replace(".pkl", ""))
    #                     epochs.append(epoch_num)
                
    #             if epochs:
    #                 latest_epoch = max(epochs)
    #                 G_AB_path = os.path.join(models_dir, f"G_AB_epoch_{latest_epoch}.pkl")
    #                 G_BA_path = os.path.join(models_dir, f"G_BA_epoch_{latest_epoch}.pkl")
    #                 D_A_path = os.path.join(models_dir, f"D_A_epoch_{latest_epoch}.pkl")
    #                 D_B_path = os.path.join(models_dir, f"D_B_epoch_{latest_epoch}.pkl")
                    
    #                 if all(os.path.exists(p) for p in [G_AB_path, G_BA_path, D_A_path, D_B_path]):
    #                     self.G_AB.load_state_dict(torch.load(G_AB_path))
    #                     self.G_BA.load_state_dict(torch.load(G_BA_path))
    #                     self.D_A.load_state_dict(torch.load(D_A_path))
    #                     self.D_B.load_state_dict(torch.load(D_B_path))
    #                     print(f"✓ Trained models loaded from epoch {latest_epoch}")
    #                 else:
    #                     raise FileNotFoundError("Not all model files found for the latest epoch")
    #             else:
    #                 raise FileNotFoundError("No trained models found")
    #         except Exception as e:
    #             print(f"Could not load trained models ({e}), initializing with random weights")
    #             self.G_AB.apply(weights_init)
    #             self.G_BA.apply(weights_init)
    #             self.D_A.apply(weights_init)
    #             self.D_B.apply(weights_init)
    #     else:
    #         # No models directory exists, initialize with random weights
    #         print(f"Models directory not found ({models_dir}), initializing with random weights")
    #         self.G_AB.apply(weights_init)
    #         self.G_BA.apply(weights_init)
    #         self.D_A.apply(weights_init)
    #         self.D_B.apply(weights_init)


