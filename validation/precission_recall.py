from torchvision.models import inception_v3, Inception_V3_Weights
import torch
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as plt
import os

from core.registries import METRICS
from torch.nn import functional as F

@METRICS.registry("prd")
class PrecisionAndRecall:
    def __init__(self,
                dataloader,
                num_samples,
                device):
        self.real_dataloader = dataloader
        self.num_samples = num_samples
        self.device = device
        #initialize the model inception v3 up to 3 conv layer
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(self.device)
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.real_features = self._initialize_real()


    @torch.inference_mode()
    def _initialize_real(self):
        all_real_features = []
        counter = 0
        for batch in self.real_dataloader:
            real = batch["x"]  
            remaining = self.num_samples - counter
            if remaining <= 0:
                break
            counter += real.shape[0]
            real = real[:remaining]
            real = real.to(self.device)
            x = self.model(self.preprocess(real))
            all_real_features.append(x.cpu())
        return torch.cat(all_real_features,dim=0)

    def preprocess(self,x):
        # GAN [-1, 1] -> [0, 1]
        x = (x + 1.0) / 2.0
        x = x.clamp(0.0, 1.0)

        # shape -> 299x299
        x = F.interpolate(
            x,
            size=(299, 299),
            mode="bilinear",
            align_corners=False,
        )
        return x

    @staticmethod
    def to_uint8(x: torch.Tensor) -> torch.Tensor:
        """
        convert GAN-images  [-1, 1] to [0, 255].
        """
        x = (x + 1.0) / 2.0
        x = x.clamp(0.0, 1.0)
        return (x * 255.0).to(torch.uint8)



    def compute(self,
                fake_data:torch.tensor,
                trainer):
        #first compute the features
        real_features = self.real_features
        #compute fake features for the given fake samples
        fake_features = self._compute_features(torch.utils.data.DataLoader(fake_data,batch_size=64,shuffle=True))

        #hier noch die shapes checken bevor das vollstädnig ist
        real_features = real_features.cpu().numpy()
        #is it faster to convert one big tensor or 2 smaller?
        fake_features = fake_features.cpu().numpy()
        #compute the clustering with k = 20 like given in the paper
        kmeans = MiniBatchKMeans(20)
        # fit the kmeans algorithm with the union of real and fake samples 
        union = np.vstack((real_features,fake_features))
        kmeans.fit(union)
        labels = kmeans.labels_
        #get real and fake samples back
        real_samples = labels[len(real_features):]
        fake_samples = labels[:len(fake_features)]
        #compute the histogram
        real_counts = np.bincount(real_samples, minlength=20)
        fake_counts = np.bincount(fake_samples, minlength=20)
        p = real_counts / real_counts.sum()
        q = fake_counts / fake_counts.sum()

        #print(them to watch them)
        
        #normalize counts
        lambdas = np.logspace(-2, 2, 100)
        alphas = []
        betas = []
        for lambd in lambdas:
            alpha = np.minimum(p / lambd, q).sum()
            beta = np.minimum(p, lambd * q).sum()
            alphas.append(alpha)
            betas.append(beta)
        alphas = np.asarray(alphas)
        betas = np.asarray(betas)
        plt.plot(alphas, betas)
        plt.xlabel("Precision")
        plt.ylabel("Recall")

        #filename and path
        path = os.path.join(trainer.project_path,"loss_plot")
        filename = f"Precision_Recall_Iteration_{trainer.num_iterations}"
        plt.savefig(os.path.join(path,filename))
        plt.close()

    @torch.inference_mode()
    def _compute_features(self,loader)->torch.tensor:
        """compute the features with the inception v3 model and returns them as a tensor

        Args:
            loader (torch.utils.data.DataLoader): DataLoader
        
        returns:
            features (torch.tensor) : computed features 
        """
        features = []
        #batchwise computation of features
        features = []
        for batch in loader:
            batch = batch.to(self.device)
            batch = self.preprocess(batch)
            x = self.model(batch)
            features.append(x.cpu())
            del batch, x
        return torch.cat(features, dim=0)

