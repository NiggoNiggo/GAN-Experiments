from torchvision.models import inception_v3, Inception_V3_Weights
import torch
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import matplotlib.pyplot as plt

class PrecissionAndRecall:
    def __init__(self,
                 real_loader
                 ):
        #initialize the model inception v3 up to 3 conv layer
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT).cuda()
        #cut the model to necessary length
        self.model = self.model[:]
        #save real features at beginning to only compute them once
        self.real_features = self._compute_features(real_loader)

    def cluster(self,
                fake_features):
        real_features = self.real_features
        fake_features = self._compute_features(self.fake_loader)

    #update function of this observer
    def compute(self,
                fake_data:torch.tensor):
        #first compute the features
        real_features = self.real_features
        #compute fake features for the given fake samples
        fake_features = self._compute_features(torch.utils.data.DataLoader(fake_data,batch_size=64,shuffle=True))
        #clustering
        #convert features to numpy format


        #hier noch die shapes checken bevor das vollstädnig ist
        real_features = real_features.cpu().numpy().transpose(1,2,0)
        #is it faster to convert one big tensor or 2 smaller?
        fake_features = fake_features.cpu().numpy().transpose(1,2,0)
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
        unique_real /= np.sum(unique_real)
        unique_fake /= np.sum(unique_fake)

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
        plt.savefig("")






        

    def _compute_features(self,loader)->torch.tensor:
        """compute the features with the inception v3 model and returns them as a tensor

        Args:
            loader (torch.utils.data.DataLoader): DataLoader
        
        returns:
            features (torch.tensor) : computed features 
        """
        features = []
        #batchwise computation of features
        for batch in loader:
            x = self.model(batch.cuda())
            features.append(x)
        #stack features to a tensor
        features = torch.stack(features,dim=1)

