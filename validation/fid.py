from torchmetrics.image.fid import FrechetInceptionDistance
import torch
import time

from torchvision.models import inception_v3

class FID:
    def __init__(self,
                 real_dataloader,
                 num_samples,
                 device=torch.device("cuda"),
                 batchsize=64):
        self.real_dataloader = real_dataloader
        self.num_samples = num_samples
        self.device = device
        self.batchsize = batchsize
        self.fid_metric = FrechetInceptionDistance(feature=2048).to(self.device)
        self.real_features = self.eval_real_data()
  

        
    @staticmethod
    def to_uint8(x):
        x = (x + 1) / 2
        x = x.clamp(0, 1)
        x = (x * 255).to(torch.uint8)
        
        return x

    def eval_real_data(self):
        current_samples = 0
        all_real = []
        for real_batch in self.real_dataloader:
            real_batch = self.to_uint8(real_batch["x"])
            current_samples += len(real_batch)
            all_real.extend(real_batch)
            remaining = self.num_samples - current_samples
            if remaining <= 0:
                break
        return torch.stack(all_real,dim=0)[:self.num_samples]


    
    def evaluate_fid(self, generator, latent_dim):
        with torch.no_grad():
            samples_generated = 0
            #update fake images
            while samples_generated < self.num_samples:
                current_batch_size = min(self.batchsize, self.num_samples - samples_generated)
                noise = torch.randn(current_batch_size, latent_dim, 1,1, device=self.device)
                fake = generator(noise)
                fake = self.to_uint8(fake)
                self.fid_metric.update(fake.to(self.device), real=False)
                samples_generated += current_batch_size
        #update real images
        for i in range(0, self.num_samples, self.batchsize):
            batch_chunk = self.real_features[i:i + self.batchsize]
            self.fid_metric.update(batch_chunk.to(self.device), real=True)
        fid_value = self.fid_metric.compute()
        self.fid_metric.reset()
        generator.train()
        return fid_value.item()
            



if __name__ == "__main__":
    start_time = time.time()
    print(torch.device("cuda"))
    from architectures.dcgan_networks import DCGANGenerator
    from torch.utils.data import DataLoader, TensorDataset
    gen = DCGANGenerator(out_shape=32, out_channels=3,latent_dim=100)
    
    noise = torch.randn(64*1000,100,1,1) #10 batches of 64 samples real
    real = torch.randn(64*1000,3,32,32)
    out = gen(noise) #fake same as above
    print(out.shape, real.shape)

    real_loader = DataLoader(TensorDataset(real), batch_size=64)
    fake_loader = DataLoader(TensorDataset(out), batch_size=64)

    #try fid 
    fid = FID(
        real_dataloader=real_loader,
        num_samples=64*1000
    )
    value = fid.evaluate_fid(gen, latent_dim=100)
    end_time = time.time()
    print("FID: ", value)
    print("Time: ", round(end_time - start_time,3))


