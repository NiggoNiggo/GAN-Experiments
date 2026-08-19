import torch 
from torch import nn
from core.registries import LOSSES

@LOSSES.registry("vanilla")
class VanillaGANLoss:
    def __init__(self,
                 label_smooting=False):
        super().__init__()
        self.loss = nn.BCELoss()
        self.label_smoothing = self.label_smoothing

    def disc_loss(self, real_pred, fake_pred):
        if self.label_smoothing:
            real_targets = torch.ones_like(real_pred)
            fake_targets = torch.zeros_like(fake_pred)
        else:
            real_targets = torch.ones_like(real_pred)*0.9
            fake_targets = torch.ones_like(fake_pred)*0.1
        loss_real = self.loss(real_pred, real_targets)
        loss_fake = self.loss(fake_pred, fake_targets)

        return loss_real + loss_fake

    def gen_loss(self, fake_pred):
        targets = torch.ones_like(fake_pred)
        return self.loss(fake_pred, targets)

        