
import os
from pathlib import Path

import torch

from .observer import Observer
import tqdm


#should save models each epoch generator and discriminator
class ModelSaver(Observer):
    def __init__(self,save_path):
        self.save_path = save_path
    
    def update(self,info):
        iterations = info["num_iterations"]
        trainer = info["trainer"]
        # Ensure model folder exists
        iteration_folder = f"iteration_{iterations}"
        models_dir = os.path.join(self.save_path,"models",iteration_folder)
        os.makedirs(models_dir, exist_ok=True)

        #save torch models
        torch.save(trainer.gen.state_dict(), os.path.join(models_dir,"Generator.pkl"))
        torch.save(trainer.disc.state_dict(), os.path.join(models_dir,"Discriminator.pkl"))
        torch.save(trainer.optim_gen.state_dict(), os.path.join(models_dir,"Optimizer_gen.pkl"))
        torch.save(trainer.optim_disc.state_dict(), os.path.join(models_dir,"Optimizer_disc.pkl"))
        #information log
        print(f"Models saved in {models_dir}")
        
