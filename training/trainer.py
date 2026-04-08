import os
import re
from pathlib import Path
import torch
from abc import ABC, abstractmethod
from tqdm.auto import tqdm
from testing.create_image import ConvGANImageSampler

from architectures.init_weights import weights_init
from organization.file_system_organizer import FileOrganizer
from observer.observer_save import ModelSaver


class GANTrainer(ABC):
    def __init__(self, 
                 gen,
                 disc, 
                 data_loader, 
                 loss_fn, 
                 optim_gen_strat, 
                 optim_disc_strat,
                 latent_dim : int,
                 save_path : str,
                 filename : str):
        """Abstract class for Training procedure of gan

        Args:
            gen (torch.nn.Module): Generator Network
            disc (torch.nn.Module): Discriminator Network
            data_loader (torch.utils.data.DataLoader): Data loader
            loss_fn (Loss_fn): Loss function
            optim_gen_strat (optim_strat): Strategy of the optimizer for generator
            optim_disc_strat (optim_strat): Strategy of the optimizer for discriminator
            latent_dim (int): Latent dimension 
        """
        self.gen = gen
        self.disc = disc 
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        #ensure that no network is None, because more complex gans like cyclegan initiate their own gens and disc and set them to zero
        if self.gen and self.disc:
            self.optim_gen = optim_gen_strat.build_optim(self.gen)
            self.optim_disc = optim_disc_strat.build_optim(self.disc)
        self.latent_dim = latent_dim
        #where the data is initilized
        self.save_path = save_path
        #observers for further functionality
        self.observers = []
        #initilaize the model whehter normal distributed or with loading a filename
        self.filename = filename
        #filename comes from search from organizer and the filesystem
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.init_project()
        
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
        for epoch in tqdm(range(epochs)):
            for batch in self.data_loader:
                d_loss, g_loss = self.train_step(batch)
            #updating the info dict with each information that is saved from here now 
            info = {"epoch":epoch,"trainer":self,"loss_d":d_loss,"loss_g":g_loss}
            self.notify(info)
            print(f"Epoch {epoch}: D={d_loss:.4f} | G={g_loss:.4f}")
            plotter = ConvGANImageSampler(self.gen, self.latent_dim)
            imgs = plotter.sample_images(64)
            plotter.plot_images_grid(imgs, num_img=64, nrow=8, normalize=True,filename=os.path.join(self.save_path,self.filename,"plots",f"epoch_{epoch}.png"))
        # return self.gen, self.disc
        #without return if you want to acces the models go for: training.gen etc.

    def init_project(self):
        file = FileOrganizer(filename=self.filename,
                             path=self.save_path)
        #creates a full build for a new project
        file.create_dir()

    def init_models(self):
        print("..... init models.....")
        if (Path(self.save_path) / self.filename).exists():
            path = Path(self.save_path) / self.filename / "models"
            last_models = []
            highest_idx = 0
            #save all files in al list
            all_files = os.listdir(path)                
            epochs = [int(re.search(r"epoch_(\d+)", f).group(1)) for f in all_files if "epoch" in f]
            if epochs:
                highest_idx = max(epochs)
            print(f"Previous Training will be continued at epoch: {highest_idx}")
            for file in all_files:
                match = re.search(fr"_epoch_{highest_idx}\.pkl", file)
                if match:
                    last_models.append(file)
            print(last_models)
            for model in last_models:
                string_in_each_model = r"_epoch_\d+.pkl"
                match = re.search(string_in_each_model,model)
                if match:
                    attr = model[:match.start()]
                    if hasattr(self,attr):
                        model_path = path / model
                        setattr(self,attr,torch.load(model_path,weights_only=True))
                        print("Loaded: ", path / model)
            print("Start training")
        else:
            print("make them normally distributed")
            if hasattr(self, "G_AB"):
                self.G_AB.apply(weights_init)
                self.G_BA.apply(weights_init)
                self.D_A.apply(weights_init)
                self.D_B.apply(weights_init)
            else:
                self.gen.apply(weights_init)
                self.disc.apply(weights_init)
                        
                        
    def attach(self,observer):
        #attach the observer to the Trainer 
        self.observers.append(observer)
        
    def detach(self,observer):
        self.observers.remove(observer)
        
    def notify(self, 
               info: dict):
        for observer in self.observers:
            #transfer self because it is the trainer instance
            observer.update(info)
        #heir vlt den trainer doch immer übergeben?