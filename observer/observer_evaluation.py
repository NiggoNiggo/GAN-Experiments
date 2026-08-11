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

# from validation.fid import FID
# from validation.precission_recall import PrecissionAndRecall

# import torch

# class Evaluate:
#     def __init__(self,
#                  num_samples:int):
#         self.eval_strats = []
#         self.num_samples = self.num_samples

#     #this is our notify
#     def compute(self,gen):
#         """compute as notify function to compute all desired metrics for evaluation

#         Args:
#             gen (torch.nn.Module): Generator network
#         """
#         #iterate through any metric
#         for metric in self.eval_strats:
#             #compute the metrics with alltogether with the same fake samples
#             print(metric.compute(self.generate_fake(gen)))

#     def attach(self,metric):
#         self.eval_strats.append(metric)

#     def generate_fake(self,gen)->torch.tensor:
#         """generate fake data for evaluation

#         Args:
#             gen (Generator) : Generator of any GAN

#         Returns:
#             fake (torch.tensor) : fake samples (num_samples)
#         """
#         #without gradients 
#         with torch.no_grad():
#             #generate num samples in one tensor 
#             noise = torch.randn(self.num_samples,100,1,1).cuda()
#             #generate fakes (num_samples,c,h,w)
#             fakes = gen(noise)
#             return fakes


#     def update_config(self,
#                       path,
#                       metrics):
#         path = os.path.join(path, "config.yaml")
#         with open(path,"r") as f:
#             data = yaml.safe_load(f)
#         if "eval" not in data:
#             data["eval"] = {}
#         # add metrics to eval category 
#         #currently only saves the last entry not the best one the logic for this will be implemented later
#         data["eval"].update(metrics)
#         #save the file again
#         with open(path,"w") as f:
#             yaml.safe_dump(data,f)
