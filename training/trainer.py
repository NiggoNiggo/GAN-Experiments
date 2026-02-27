from abc import ABC, abstractmethod

class GANTrainer(ABC):
    def __init__(self, gen, disc, data_loader, loss_fn, optim_gen, optim_disc):
        self.gen = gen
        self.disc = disc 
        self.data_loader = data_loader
        self.loss_fn = loss_fn
        self.optim_gen = optim_gen
        self.optim_disc = optim_disc
        
    @abstractmethod
    def train_disc(self):
        pass
    @abstractmethod
    def train_gen(self):
        pass
    @abstractmethod
    def train_step(self):
        pass
    
    def train(self, epochs):
        for epoch in range(epochs):
            for batch in self.data_loader:
                d_loss, g_loss = self.train_step(batch)

            print(f"Epoch {epoch}: D={d_loss:.4f} | G={g_loss:.4f}")