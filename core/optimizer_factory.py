from itertools import chain


class OptimizerStrategy:
    def build_optim(self,model):
        raise NotImplementedError()
    


class AdamStrategy(OptimizerStrategy):
    def __init__(self,lr=0.0002,betas=(0.5,0.999)):
        self.lr = lr
        self.betas = betas
        
    def build_optim(self,model):
        import torch.optim as optim
        return optim.Adam(model.parameters(),lr=self.lr,betas=self.betas)
    
