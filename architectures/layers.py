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
    def __init__(self,input_dim,output_dim,kernel_size,stride,padding,batch_norm:bool=False,last_layer:bool=False):
        super().__init__()

        self.model = [
            nn.Conv2d(in_channels=input_dim,out_channels=output_dim,kernel_size=kernel_size,stride=stride,padding=padding,bias=False)
            ]
        if batch_norm and last_layer is False:
            self.model.append(nn.BatchNorm2d(output_dim))
        if not last_layer:
            self.model.append(nn.LeakyReLU(0.2,inplace=True))
        
        self.model = nn.Sequential(*self.model)
        
        
        
    def forward(self,x):
        return self.model(x)

class ConvTransposeLayer(nn.Module):
    def __init__(self,input_dim,output_dim,kernel_size,stride,padding,last_layer:bool=True):
        super().__init__()

        self.model = [
            nn.ConvTranspose2d(in_channels=input_dim,out_channels=output_dim,kernel_size=kernel_size,stride=stride,padding=padding,bias=False)
        ]
        if last_layer:
            self.model.append(nn.Tanh())
        else:
            self.model.append(nn.BatchNorm2d(output_dim)),
            self.model.append(nn.ReLU(True))
            
        self.model = nn.Sequential(*self.model)
        
        
    def forward(self,x):
        return self.model(x)