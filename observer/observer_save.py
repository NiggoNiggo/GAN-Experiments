
import os, re
import torch
import pandas as pd

from .observer import Observer


#should save models each epoch generator and discriminator
class ModelSaver(Observer):
    def __init__(self,save_path):
        self.save_path = save_path
    
    def update(self,info):
        iterations = info["num_iterations"]
        trainer = info["trainer"]

        #get csv file through the csv value


        #check if current fid is smallest compared to previous models
        fid_col = self.read_csv_values()
        # is_best = self.check_if_best(fid_score)
        #only uses FID to compare models
        is_best = True if fid_col.iloc[-1] < fid_col.iloc[:-1].min() else False
        if is_best:
            # Ensure model folder exists
            fid_score = fid_col.iloc[-1]
            iteration_folder = f"iteration_{iterations}_fid_{fid_score:.2f}"
            models_dir = os.path.join(self.save_path,"models",iteration_folder)
            os.makedirs(models_dir, exist_ok=True)
            #save torch models
            torch.save(trainer.gen.state_dict(), os.path.join(models_dir,"Generator.pkl"))
            torch.save(trainer.disc.state_dict(), os.path.join(models_dir,"Discriminator.pkl"))
            torch.save(trainer.optim_gen.state_dict(), os.path.join(models_dir,"Optimizer_gen.pkl"))
            torch.save(trainer.optim_disc.state_dict(), os.path.join(models_dir,"Optimizer_disc.pkl"))
            #information log
            print(f"Models saved in {models_dir}")

    # def check_if_best(self,
    #                   current_fid:float):
    #     all_scores = []
    #     #search for all dirs in the models folder 
    #     pattern = r"iteration_\d+_fid_(\d+\.\d{2})$"
    #     for r,d,f in os.walk(os.path.join(self.save_path,"models")):
    #         for dire in d:
    #             match = re.search(pattern,dire,re.IGNORECASE)
    #         all_scores.append(float(match.group(0)))
    #     #find min score
    #     min_fid = min(all_scores)
    #     #if current is smaller than minimal return True
    #     if current_fid < min_fid:
    #         return True
    #     #return false if is higher than the minimal fid measures until now
    #     return False

    def read_csv_values(self):
        fid_col = pd.read_csv(os.path.join(self.save_path,"values_csv","values.csv"),index_col=False)
        fid_col = fid_col["FID"]
        return fid_col
            
            
        
        
