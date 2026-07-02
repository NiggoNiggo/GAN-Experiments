import torch 
from torch import nn

class HingeLoss:
    def __init__(self):
        super().__init__()
    
    def disc_loss(self,real_pred,fake_pred):
        loss = torch.relu(1-real_pred).mean() + torch.relu(1+fake_pred).mean()
        return loss


    def gen_loss(self,fake_pred):
        gen_loss = -torch.mean(fake_pred)
        return gen_loss