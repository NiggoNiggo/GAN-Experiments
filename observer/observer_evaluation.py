from validation.fid import FID
from observer.observer import Observer



class EvalObserver(Observer):
    def __init__(self):
        pass
        #hier other metrics

    
    def update(self,info):
        if info["epoch"] % 5 == 0:
            trainer = info['trainer']
            num_samples = 10000 if len(trainer.data_loader.dataset) >= 10000 else len(trainer.data_loader.dataset)
            fid_eval = FID(trainer.data_loader, num_samples, trainer.device, trainer.data_loader.batch_size)
            gen = trainer.gen
            latent_dim = trainer.latent_dim
            epoch = trainer.epoch
            fid_value = fid_eval.evaluate_fid(gen, latent_dim)
            print(f"FID: {fid_value}, Epoch: {epoch}")
            info["fid"] = fid_value
    
    def eval_fid(self,gen, latent_dim):
        pass

    def eval_fad(self, gen, latent_dim):
        pass