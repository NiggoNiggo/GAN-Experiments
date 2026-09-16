import torch
from torch import nn

class RelativisticLoss:
    def __init__(self):
        self.bce = nn.BCELoss()

    def disc_loss(self,real,fake):
        loss = real- fake

    def gen_loss(self):
        pass