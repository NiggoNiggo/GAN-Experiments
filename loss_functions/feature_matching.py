from core.registries import LOSSES
import torch


@LOSSES.registry("feature_matching")
class FeatureMatching:
    def __init__(self,
                 label_smoothing=False):
        """Feature matching algorithm to regularize training and avoid mode collapse

        Args:
            label_smoothing (bool, optional): if label smoothing should be applied. Defaults to False.
        """
        super().__init__()
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.label_smoothing = label_smoothing

    def disc_loss(self,
                  real_pred,
                  fake_pred):
        """Discriminator loss for feature matching loss function 

        Args:
            real_pred (torch.tensor): Discriminator predictions for real
            fake_pred (torch.tensor): Discriminator predictions for fake

        Returns:
            torch.tensor: Discriminator loss 
        """
        if self.label_smoothing:
            #like said in the paper only smooth ne the positive not the nefative
            real_targets = torch.ones_like(real_pred)*0.9
            fake_targets = torch.zeros_like(fake_pred)
        else:
            real_targets = torch.ones_like(real_pred)
            fake_targets = torch.zeros_like(fake_pred)
        loss_real = self.loss(real_pred, real_targets)
        loss_fake = self.loss(fake_pred, fake_targets)

        return loss_real + loss_fake

    def gen_loss(self,
                 real_features, 
                 fake_features,
                 real_pred,
                 fake_pred):
        """

        Args:
            real_features (torch.tensor): features of disc intermediate layer  for real data
            fake_features (torch.tensor): features of disc intermediate layer  for fake data

        Returns:
            torch.tensor: Generator loss 
        """
        real_mean = torch.mean(real_features,dim=0)
        fake_mean = torch.mean(fake_features,dim=0)
        #additional adversarial loss
        feature_loss = torch.mean((real_mean - fake_mean) ** 2)
        # adv_loss = self.loss(fake_pred,torch.ones_like(fake_pred))
        # return adv_loss + 1 * feature_loss
        return feature_loss