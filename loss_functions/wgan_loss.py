import torch


class WGANLoss():
    def __init__(self):
        pass
    def disc_loss(self,real_preds,fake_preds):
        return fake_preds.mean() - real_preds.mean()
    
    def gen_loss(self,fake_preds):
        return - fake_preds.mean()