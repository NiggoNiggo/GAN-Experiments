from core.registries import LOSSES
import torch


@LOSSES.registry("feature_matching")
class FeatureMatching:
    def __init__(self,
                 label_smoothing=False):
        super().__init__()
        self.loss = torch.nn.BCELoss()
        self.label_smoothing = label_smoothing

    def disc_loss(self, real_pred, fake_pred):
        if self.label_smoothing:
            #like said in the paper only smooth ne the positive not the nefative
            real_targets = torch.ones_like(real_pred)*0.9
            fake_targets = torch.zeros_like(fake_pred)
        else:
            real_targets = torch.ones_like(real_pred)
            fake_targets = torch.ones_like(fake_pred)
        loss_real = self.loss(real_pred, real_targets)
        loss_fake = self.loss(fake_pred, fake_targets)

        return loss_real + loss_fake

    def gen_loss(self,
                 real_features, 
                 fake_features):
        real_mean = torch.mean(real_features,dim=0)
        fake_mean = torch.mean(fake_features,dim=0)
        return torch.mean((real_mean-fake_mean)**2)