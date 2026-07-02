import torch 
from torch import nn

class LSLoss:
    def __init__(self):
        super().__init__()
    
    def disc_loss(self,real_pred,fake_pred):
        real_targets = torch.ones_like(real_pred)

        fake_targets = torch.zeros_like(fake_pred)

        loss_real = 0.5*torch.mean((real_pred - real_targets)**2)
        loss_fake = 0.5*torch.mean((fake_pred-fake_targets)**2)
        return loss_real + loss_fake


    def gen_loss(self,fake_pred):
        gen_loss = 0.5*torch.mean((fake_pred-torch.ones_like(fake_pred))**2)
        return gen_loss