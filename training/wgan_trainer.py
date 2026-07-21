from trainer import GANTrainer
class WGANTrainer(GANTrainer):


    def __init__(self, 
                gen, 
                disc, 
                data_loader, 
                loss_fn, 
                optim_gen_strat, 
                optim_disc_strat, 
                latent_dim,save_path,
                filename,
                device="cuda",
                mode="clipping"):
        super().__init__(gen, disc, data_loader, loss_fn, optim_gen_strat, optim_disc_strat,latent_dim,save_path,filename)
        if mode == "clipping":
            self.penalty = self.apply_clipping()
        else:
            self.penalty = self.apply_interpolation()
    
    def train_step(self, batch):
        #TODO train loop
        # TODO train disc 5 times
        # TODO train gen once
        return 
    
    def train_gen(self):
        return 
    
    def train_disc(self):
        return 

    def apply_clipping(self):
        # TODO receivs the gradients annd clip [-1,1]
        # TODO return the gradients or apply them inplace
        return 
    
    def apply_interpolation(self):
        return