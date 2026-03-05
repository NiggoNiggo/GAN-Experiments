from abc import ABC, abstractmethod
from tqdm.auto import tqdm
from testing.create_image import SampleImages
from architectures.init_weights import weights_init
import torch


class GANTrainer(ABC):
    def __init__(self, gen, disc, data_loader, loss_fn, optim_gen_strat, optim_disc_strat,latent_dim):
        self.gen = gen
        self.disc = disc 
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.optim_gen = optim_gen_strat.build_optim(self.gen)
        self.optim_disc = optim_disc_strat.build_optim(self.disc)
        self.latent_dim = latent_dim

        self.init_models()
        
    @abstractmethod
    def train_disc(self):
        pass
    @abstractmethod
    def train_gen(self):
        pass
    @abstractmethod
    def train_step(self,batch):
        pass
    
    def ensure_correct_input(self,batch):
        real = batch["x"].to(self.device)
        labels = batch.get("y")  
        if labels is not None:
            labels = labels.to(self.device)
        batch_size = real.size(0)
        return batch_size, real, labels
    
    def train(self, epochs):
        self.image_plotter = SampleImages(self.gen,self.latent_dim)
        for epoch in tqdm(range(epochs)):
            for batch in self.data_loader:
                d_loss, g_loss = self.train_step(batch)

            print(f"Epoch {epoch}: D={d_loss:.4f} | G={g_loss:.4f}")
            if epoch % 10 == 0:
                self.image_plotter.plot_images_grid(16)
        return self.gen, self.disc

    def init_models(self,**filenames):
        if filenames:
            self.gen.load_state_dict(torch.load(filenames["gen"]))
            self.disc.load_state_dict(torch.load(filenames["disc"]))
        else:
            self.gen.apply(weights_init)
            self.disc.apply(weights_init)
                
        