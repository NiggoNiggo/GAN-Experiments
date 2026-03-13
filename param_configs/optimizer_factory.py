from itertools import chain


class OptimizerStrategy:
    def buil_optim(self,model):
        raise NotImplementedError()
    


class AdamStrategy(OptimizerStrategy):
    def __init__(self,lr=0.0002,betas=(0.5,0.999)):
        self.lr = lr
        self.betas = betas
        
    def build_optim(self,model):
        import torch.optim as optim
        return optim.Adam(model.parameters(),lr=self.lr,betas=self.betas)
    
class CycleStrategy(OptimizerStrategy):
    def __init__(self,lr=0.0002,betas=(0.5,0.999)):
        self.lr = lr
        self.betas = betas
    
    def build_optim(self,model1,model2):
        import torch.optim as optim
        return optim.Adam(list(model1.parameters())+list(model2.parameters()),lr=self.lr,betas=self.betas)