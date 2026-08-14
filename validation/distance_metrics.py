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

        # Real features werden nach dem ersten update behalten.
        # reset() löscht später nur die Fake-Statistiken.
        self.fid = FrechetInceptionDistance(
            reset_real_features=False,
        ).to(self.device)

        self.kid = KernelInceptionDistance(reset_real_features=False).to(self.device)

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
        fid_value = self.fid.compute()
        kid_value = self.kid.compute()
        kid_value = round(kid_value[0].item(),3)
        fid_value = round(fid_value.item(),3)
        print(f"FID: {fid_value:.3f} \t KID: {kid_value:.3f}")
        self.update_csv(trainer,fid_value,kid_value)
        self.fid.reset()
        self.kid.reset()
        return {
                "fid": fid_value,
                "kid": kid_value,
            }

    def update_csv(self, trainer, fid, kid):
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
        data.to_csv(path, index=False)


        
            



# if __name__ == "__main__":
#     start_time = time.time()
#     print(torch.device("cuda"))
#     from architectures.dcgan_networks import DCGANGenerator
#     from torch.utils.data import DataLoader, TensorDataset
#     gen = DCGANGenerator(out_shape=32, out_channels=3,latent_dim=100)
    
#     noise = torch.randn(64*1000,100,1,1) #10 batches of 64 samples real
#     real = torch.randn(64*1000,3,32,32)
#     out = gen(noise) #fake same as above
#     print(out.shape, real.shape)

#     real_loader = DataLoader(TensorDataset(real), batch_size=64)
#     fake_loader = DataLoader(TensorDataset(out), batch_size=64)

#     #try fid 
#     fid = FID(
#         real_dataloader=real_loader,
#         num_samples=64*1000
#     )
#     value = fid.evaluate_fid(gen, latent_dim=100)
#     end_time = time.time()
#     print("FID: ", value)
#     print("Time: ", round(end_time - start_time,3))


