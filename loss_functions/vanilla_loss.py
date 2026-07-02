import torch 
from torch import nn

class VanillaGANLoss:
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()

    def disc_loss(self, real_pred, fake_pred):
        real_targets = torch.ones_like(real_pred)*0.9
        fake_targets = torch.zeros_like(fake_pred)*0.9

        loss_real = self.loss(real_pred, real_targets)
        loss_fake = self.loss(fake_pred, fake_targets)

        return loss_real + loss_fake

    def gen_loss(self, fake_pred):
        targets = torch.ones_like(fake_pred)
        return self.loss(fake_pred, targets)

        