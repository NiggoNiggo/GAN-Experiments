
import os

import torch

from .observer import Observer


#should save models each epoch generator and discriminator
class ModelSaver(Observer):
    def __init__(self,save_path):
        self.save_path = save_path
    
    def update(self,info):
        valid_keys = ["trainer","epoch","metric"]
        epoch = info["epoch"]
        filename = f"epoch_{epoch}.pkl"
        #save torch models
        torch.save(info["trainer"].gen.state_dict(),os.path.join(self.save_path,"models","Generator_" +filename))
        torch.save(info["trainer"].disc.state_dict(),os.path.join(self.save_path,"models","Discriminator_"+filename))
        #information log
        print(f"Generator and Discriminator saved in {self.save_path}")