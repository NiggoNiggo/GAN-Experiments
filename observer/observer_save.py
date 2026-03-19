
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
        trainer = info["trainer"]
        # Ensure model folder exists
        models_dir = os.path.join(self.save_path, "models")
        os.makedirs(models_dir, exist_ok=True)

        #save torch models
        try:
            if hasattr(trainer, 'G_AB'):  # Check if it's a CycleGANTrainer
                torch.save(trainer.G_AB.state_dict(), os.path.join(models_dir, "G_AB_" + filename))
                torch.save(trainer.G_BA.state_dict(), os.path.join(models_dir, "G_BA_" + filename))
                torch.save(trainer.D_A.state_dict(), os.path.join(models_dir, "D_A_" + filename))
                torch.save(trainer.D_B.state_dict(), os.path.join(models_dir, "D_B_" + filename))
            else:
                torch.save(trainer.gen.state_dict(), os.path.join(models_dir, "Generator_" + filename))
                torch.save(trainer.disc.state_dict(), os.path.join(models_dir, "Discriminator_" + filename))
            #information log
            print(f"Models saved in {models_dir}")
        except Exception as e:
            print(f"ModelSaver failed: {e}")
            # Don't stop training because of observer errors
