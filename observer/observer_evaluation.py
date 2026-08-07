from validation.fid import FID
from observer.observer import Observer
import yaml, os
import torch



class EvalObserver(Observer):
    def __init__(self):
        pass
        #hier other metrics

    
    def update(self,info):
        #access data from info object received from the trainer base class
        trainer = info['trainer']
        num_iters = info["num_iterations"]
        #define the amount of data for computation of FID
        num_samples = 10000 if len(trainer.data_loader.dataset) >= 10000 else len(trainer.data_loader.dataset)
        #eval FID
        fid_eval = FID(trainer.data_loader, num_samples, trainer.device, trainer.data_loader.batch_size)
        gen = trainer.gen
        latent_dim = trainer.latent_dim
        #computed the actual FID with round 4 digits
        fid_value = round(fid_eval.evaluate_fid(gen, latent_dim),4)
        #write information to console for tracking
        print(f"num_iterations: {num_iters} | FID: {fid_value}")
        info["fid"] = fid_value
        #get gpu usage 
        #save the data to the yaml file in the project folder for better reproduceability
        write_data = {"num_iterations":num_iters,
                      "FID":fid_value}

        self.update_config(trainer.project_path,write_data)

    def update_config(self,
                      path,
                      metrics):
        path = os.path.join(path, "config.yaml")
        with open(path,"r") as f:
            data = yaml.safe_load(f)
        if "eval" not in data:
            data["eval"] = {}
        # add metrics to eval category 
        #currently only saves the last entry not the best one the logic for this will be implemented later
        data["eval"].update(metrics)
        #save the file again
        with open(path,"w") as f:
            yaml.safe_dump(data,f)

    
    def eval_fid(self,gen, latent_dim):
        pass

    def eval_fad(self, gen, latent_dim):
        pass

    def eval_inception_score(self):
        pass