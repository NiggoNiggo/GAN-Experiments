import os
import re
from core.registries import *
from pathlib import Path
import torch
from abc import ABC, abstractmethod
from tqdm.auto import tqdm
from testing.create_image import ConvGANImageSampler
import torchvision.transforms as T

from architectures.init_weights import weights_init
from organization.file_system_organizer import FileOrganizer
from observer.observer_save import ModelSaver
from data.wrappers import DataWrapper
from core.optimizer_factory import AdamStrategy

#importall the modules that are registred
import architectures, loss_functions, data


class GANTrainer(ABC):
    def __init__(self, 
                 config
               ):

        self.cfg = config
        #ensure that no network is None, because more complex gans like cyclegan initiate their own gens and disc and set them to zero
        #make the optimizer, but this should be more clearer in future
        
        #observers for further functionality
        self.observers = []
        #initilaize the model whehter normal distributed or with loading a filename
        self.filename = self.cfg["training"]["args"]["project_name"]
        self.save_path = self.cfg["training"]["args"]["save_path"]
        self.latent_dim = self.cfg["params"]["latent_dim"]
        #filename comes from search from organizer and the filesystem
        self.device = self._get_device()
        self.encode_config()
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
        for epoch in range(self.epoch, self.epoch+epochs):
            self.epoch = epoch
            for batch in tqdm(self.data_loader):
                d_loss, g_loss = self.train_step(batch)
            #updating the info dict with each information that is saved from here now 
            info = {"epoch":epoch,"trainer":self,"loss_d":d_loss,"loss_g":g_loss}
            #maybe change the info to execute it each iteration not only after epoch
            #allows better comparability with different models
            self.notify(info)
            print(f"Epoch {epoch}: D={d_loss:.4f} | G={g_loss:.4f}")

    


    def init_project(self):
        file = FileOrganizer(filename=self.filename,
                             path=self.save_path)
        print(f"Project located in {os.path.join(self.save_path,self.filename)}")
        #creates a full build for a new project
        file.create_dir()

    def init_models(self):
        print("..... init models.....")
        if len(os.listdir((Path(self.save_path) / self.filename))) > 1:
            path = Path(self.save_path) / self.filename / "models"
            last_models = []
            highest_idx = 0
            #save all files in al list
            all_files = os.listdir(path)                
            epochs = [int(re.search(r"epoch_(\d+)", f).group(1)) for f in all_files if "epoch" in f]
            if epochs:
                highest_idx = max(epochs)
                self.epoch = highest_idx + 1
            else:
                self.epoch = 1
            print(f"Previous Training will be continued at epoch: {highest_idx}")
            for file in all_files:
                match = re.search(fr"_epoch_{highest_idx}\.pkl", file)
                if match:
                    last_models.append(file)
            for model in last_models:
                string_in_each_model = r"_epoch_\d+.pkl"
                match = re.search(string_in_each_model, model)
                if match:
                    attr = model[:match.start()]
                    if attr == "Generator": attr = "gen"    
                    if attr == "Discriminator": attr = "disc"    

                    if hasattr(self, attr):
                        model_path = path / model

                        state_dict = torch.load(model_path, weights_only=True)

                        model_obj = getattr(self, attr)
                        model_obj.load_state_dict(state_dict)

                        print("Loaded: ", model_path)

            print("Start training")
        else:
            print("make them normally distributed")
            self.gen.apply(weights_init)
            self.disc.apply(weights_init)
            self.epoch = 1

    def sample_images(self, num_img=64):
        imgs = self.plotter.sample_images(num_img)
        self.plotter.plot_images_grid(imgs, num_img=num_img, nrow=8, normalize=True,filename=os.path.join(self.save_path,self.filename,"plots",f"epoch_{self.epoch}.png"))
                        
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
    
    def _get_device(self):
        """Allocate the device for usage. Tries to find a gpu, if no gpu found the device is set to cpu. However, the training is not recommendended with only cpu. Watch your dataset size and the corresponding models.

        Returns:
            torch.cuda.device: GPU if found else CPU
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using device: {torch.cuda.get_device_name(device)}")
        else:
            device = torch.device("cpu")
            print(f"Using  device: {device}, not recommendend")
        return device

    def encode_config(self):
        #load the generator
        gen_cls = GENERATORS.get(self.cfg["generator"]["name"])
        self.gen = gen_cls(**self.cfg["generator"]["args"])
        self.gen = self.gen.to(self.device)

        #load the disc:
        disc_cls = DISCRIMINATORS.get(self.cfg["discriminator"]["name"])
        self.disc = disc_cls(**self.cfg["discriminator"]["args"])
        self.disc = self.disc.to(self.device)

        #load the loss function
        loss = LOSSES.get(self.cfg["loss"]["name"])
        self.loss_fn = loss()
        
        #load the data
        transform = T.Compose([
        T.Resize((self.cfg["training"]["args"]["out_shape"], 
                        self.cfg["training"]["args"]["out_shape"])),      
        T.ToTensor(),             
        T.Normalize((0.5,), (0.5,), (0.5,))  
        ])

        #prepare dataset:
        data_cls = DATASETS.get(self.cfg["dataset"]["name"])
        dataset = data_cls(self.cfg["dataset"]["data_path"],transform)

        #make dataloader ready
        self.data_loader=torch.utils.data.DataLoader(
            DataWrapper(dataset,
                        has_labels=self.cfg["params"]["has_labels"]
                        ),
                        batch_size=self.cfg["params"]["batchsize"],
                        shuffle=True,
                        pin_memory=True,
                        num_workers=self.cfg["params"]["num_workers"],persistent_workers=True)

        self.optim_gen=AdamStrategy(lr=self.cfg["params"]["lr_gen"], betas=(0.5, 0.999)).build_optim(self.gen)
        self.optim_disc=AdamStrategy(lr=self.cfg["params"]["lr_disc"], betas=(0.5, 0.999)).build_optim(self.disc)


        