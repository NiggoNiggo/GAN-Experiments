from torch import nn
class LinearLayer(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LinearLayer,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim,output_dim),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self,x):
        return self.model(x)