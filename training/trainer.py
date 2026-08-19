import os
import re
import yaml
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
import architectures, loss_functions, data, observer, validation


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
        self.project_path = os.path.join(self.save_path,self.filename)
        self.latent_dim = self.cfg["params"]["latent_dim"]
        #filename comes from search from organizer and the filesystem
        self.device = self._get_device()
        self.encode_config()
        self.init_project()
        self.get_params()
        
    
    
    @abstractmethod
    def train_disc(self):
        pass
    @abstractmethod
    def train_gen(self):
        pass
    @abstractmethod
    def train_step(self,
                   batch):
        pass
    
    def ensure_correct_input(self,batch):
        real = batch["x"].to(self.device)
        labels = batch.get("y")  
        if labels is not None:
            labels = labels.to(self.device)
        batch_size = real.size(0)
        return batch_size, real, labels
    
    def train(self):
        train_duration = range(self.num_iterations, self.num_iterations + self.cfg["params"]["iterations"])
        print(f"Training for {len(train_duration)} iterations")
        for epoch in train_duration:
            self.epoch = epoch
            pbar = tqdm(self.data_loader)
            for batch in pbar:
                mem = torch.cuda.memory_reserved(self.device) / 1024**2
                pbar.set_postfix(memory=f"{mem:.1f} MB")
                d_loss, g_loss = self.train_step(batch)
                self.num_iterations += 1
                #like for each epoch change to a comparable amount of batches computed
                if self.num_iterations % self.cfg["params"]["logging_iterations"] == 0 and self.num_iterations > 0:
                    #call the info to notify the observers to do their things
                    
                    info = {"num_iterations":self.num_iterations,"trainer":self,"loss_d":round(d_loss,6),"loss_g":round(g_loss,6)}
                    self.notify(info)
                    #just print some informations every 1000 Iterations
                    tqdm.write(f"Iterations {self.num_iterations}: D={d_loss:.4f} | G={g_loss:.4f}")
            

    def init_project(self):
        file = FileOrganizer(filename=self.filename,
                             path=self.save_path)
        path = os.path.join(self.save_path,self.filename)
        print(f"Project located in {path}")
        #creates a full build for a new project
        file.create_dir()
        #save the config file
        with open(os.path.join(path,"config.yaml"),"w") as f:
            yaml.safe_dump(self.cfg,f,sort_keys=False)



    def init_models(self):
        print("..... init models.....")
        models_root = Path(self.save_path) / self.filename / "models"
        if not models_root.exists():
            self.epoch = 1
            print("Beginn training von Epoch 1....")
        # search for highest iteration saved 
        epoch_dirs = []
        for folder in models_root.iterdir():
            if folder.is_dir():
                match = re.match(r"iteration_(\d+)k", folder.name)
                if match:
                    epoch_dirs.append((int(match.group(1)), folder))
        if not epoch_dirs:
            # self.epoch = 1
            self.num_iterations = 0
            return
        # find highest epoch
        highest_epoch, latest_folder = max(epoch_dirs, key=lambda x: x[0])
        self.epoch = highest_epoch + 1
        print(f"Previous training will be continued at iteration {highest_epoch}")
        self.num_iterations = highest_epoch
        print("Training continoued Iteration: ", self.num_iterations)
        # load models
        for model_file in latest_folder.glob("*.pkl"):
            name = model_file.stem
            if name == "Generator":
                attr = "gen"
            elif name == "Discriminator":
                attr = "disc"
            else:
                continue
            if hasattr(self, attr):
                state_dict = torch.load(model_file, weights_only=True)
                getattr(self, attr).load_state_dict(state_dict)
                print(f"Loaded: {model_file}")
                print("Start training")
            else:
                print("Initialize models normally distributed")
                self.gen.apply(weights_init)
                self.disc.apply(weights_init)
                self.num_iterations = 0
        

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
        #label smooting preferable only at normal gan loss with bce 
        label_smoothing = self.cfg["loss"]["label_smoothing"]
        #apply label smooting
        if label_smoothing:
            self.loss_fn = loss(True)
        else:
            self.loss_fn = loss(False)
        
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

        self.optim_gen=AdamStrategy(lr=self.cfg["params"]["lr_gen"], 
                                    betas=(self.cfg["params"]["beta1"], 
                                           self.cfg["params"]["beta2"])
                                           ).build_optim(self.gen)

        self.optim_disc=AdamStrategy(lr=self.cfg["params"]["lr_disc"], 
                                     betas=(self.cfg["params"]["beta1"],
                                             self.cfg["params"]["beta2"])
                                             ).build_optim(self.disc)

    def get_params(self):
        gen_params = sum([p.numel() for p in self.gen.parameters()])
        disc_params = sum([p.numel() for p in self.disc.parameters()])
        print(f"Generator parameters: {gen_params} | Discriminator parameters: {disc_params}")

    
        