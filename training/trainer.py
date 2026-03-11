import torch
from abc import ABC, abstractmethod
from tqdm.auto import tqdm

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
        self.init_models()
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
        return self.gen, self.disc

    def init_project(self):
        file = FileOrganizer(filename=self.filename,
                             path=self.save_path)
        #creates a full build for a new project
        file.create_dir()

    def init_models(self,**filenames):
        if filenames:
            self.gen.load_state_dict(torch.load(filenames["gen"]))
            self.disc.load_state_dict(torch.load(filenames["disc"]))
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
        