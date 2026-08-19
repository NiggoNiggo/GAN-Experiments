import os
import yaml
import torch

from core.registries import METRICS

class Evaluate:
    def __init__(self,
                 dataloader,
                 device):
        self.real_loader = dataloader
        self.num_samples = min(5000, len(dataloader.dataset))
        #Cuda
        self.device = device
        # Metric instances
        self.metrics = {}
        # Whether metrics have already been initialized
        self.initialized = False

    #function to register the metrics from the config file
    def _initialize_metrics(self, cfg):
        if self.initialized:
            return
        # registry all metrics to compute them sequentially
        for metric_name, registry_name in cfg["metrics"].items():
            metric_cls = METRICS.get(registry_name)
            #every metric class receives num_samples and device and real_loader
            self.metrics[metric_name] = metric_cls(dataloader=self.real_loader,
                                                   num_samples=self.num_samples,
                                                   device=self.device)
        #True thus this procedure must not be repeated 
        self.initialized = True


    def update(self, info):
        #called by GANTrainer notify
        trainer = info["trainer"]
        cfg = trainer.cfg
        # Initialize metrics once
        self._initialize_metrics(cfg)
        fake_data = self.generate_fakes(
                                    gen=trainer.gen,
                                    latent_dim=trainer.latent_dim,
                                    )

        results = {
            "iteration": info["num_iterations"],
            "loss_d": info["loss_d"],
            "loss_g": info["loss_g"],
        }
        #compute the metrics and save it to the results
        for name, metric in self.metrics.items():
            value = metric.compute(fake_data,trainer)
            results[name] = value





    @torch.inference_mode()
    def generate_fakes(self, gen, latent_dim):
        all_fakes = []
        for _ in range(0, self.num_samples, 256):
            current_bs = min(256,self.num_samples - sum(x.shape[0] for x in all_fakes))
            noise = torch.randn(current_bs,latent_dim,1,1,device=self.device)
            fake = gen(noise)
            all_fakes.append(fake.cpu())
            del noise, fake
        return torch.cat(all_fakes, dim=0)


