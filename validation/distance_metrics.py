from torchmetrics.image.fid import FrechetInceptionDistance
import torch
import time
import os
import pandas as pd

from core.registries import METRICS

import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore

from core.registries import METRICS


@METRICS.registry("fid")
class InceptionDistanceMeasures:

    def __init__(
        self,
        dataloader,
        num_samples: int,
        device,
        batch_size: int = 128,
    ):
        self.real_dataloader = dataloader
        self.num_samples = num_samples
        self.device = device
        self.batch_size = batch_size

        #frechèt Inception Score
        self.fid = FrechetInceptionDistance(
            reset_real_features=False,
        ).to(self.device)
        #kernel inceotion score 
        self.kid = KernelInceptionDistance(reset_real_features=False).to(self.device)
        #Inception Score
        self.is_score = InceptionScore(normalize=False).to(self.device)

        # Real-Daten genau EINMAL durch Inception schicken.
        self._initialize_real()

    @staticmethod
    def to_uint8(x: torch.Tensor) -> torch.Tensor:
        """
        Konvertiert GAN-Bilder von [-1, 1] nach [0, 255].
        """
        x = (x + 1.0) / 2.0
        x = x.clamp(0.0, 1.0)
        return (x * 255.0).to(torch.uint8)

    @torch.inference_mode()
    def _initialize_real(self):
        counter = 0
        for batch in self.real_dataloader:
            real = batch["x"]  
            remaining = self.num_samples - counter
            if remaining <= 0:
                break
            counter += real.shape[0]
            real = real[:remaining]
            real = real.to(self.device, non_blocking=True)
            real = self.to_uint8(real)

            self.fid.update(real, real=True)
            self.kid.update(real, real=True)



    @torch.inference_mode()
    def compute(self,
                fake:torch.Tensor,
                trainer
                ) -> float:
        """
        Compute FID

        Args:
            fake:
                Tensor mit Shape [N, C, H, W]
                und Werten im Bereich [-1, 1].
        Additional Args:
            path : str
        Returns:
            FID score als float.
        """
        fake_loader = DataLoader(fake,batch_size=self.batch_size,shuffle=False)
        for batch in fake_loader:
            batch = batch.to(self.device,non_blocking=True)
            batch = self.to_uint8(batch)
            self.fid.update(batch,real=False)
            self.kid.update(batch,real=False)
            self.is_score.update(batch)
        fid_value = self.fid.compute()
        kid_value = self.kid.compute()
        is_value = self.is_score.compute()
        kid_value = round(kid_value[0].item(),3)
        fid_value = round(fid_value.item(),3)
        is_value = round(is_value[0].item(),3)
        print(f"FID: {fid_value:.3f} |\t KID: {kid_value:.3f} |\t IS: {is_value:.3f}")
        self.update_csv(trainer,fid_value,kid_value,is_value)
        self.fid.reset()
        self.kid.reset()
        self.is_score.reset()
        return {
                "fid": fid_value,
                "kid": kid_value,
                "is": is_value
            }

    def update_csv(self,
                   trainer,
                   fid:float,
                   kid:float,
                   is_score:float):
        path = os.path.join(
            trainer.project_path,
            "values_csv",
            "values.csv"
        )
        data = pd.read_csv(path)
        iteration = trainer.num_iterations
        mask = data["num_iterations"] == iteration
        data.loc[mask, "FID"] = fid
        data.loc[mask, "KID"] = kid
        data.loc[mask, "IS"] = is_score
        data.to_csv(path, index=False)


        
            


