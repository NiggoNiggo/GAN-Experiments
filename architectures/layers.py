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


class ConvLayer(nn.Module):
    def __init__(self,input_dim,output_dim,kernel_size,stride,padding):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=input_dim,out_channels=output_dim,kernel_size=kernel_size,stride=stride,padding=padding),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(0.2)
        )
        
        
    def forward(self,x):
        return self.model(x)

class ConvTransposeLayer(nn.Module):
    def __init__(self,input_dim,output_dim,kernel_size,stride,padding):
        super().__init__()

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=input_dim,out_channels=output_dim,kernel_size=kernel_size,stride=stride,padding=padding),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(True)
        )
        
        
    def forward(self,x):
        return self.model(x)