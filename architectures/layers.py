from torch import nn
from core.registries import BLOCKS
import torch.nn.functional as F
import torch


@BLOCKS.registry("linear")
class LinearLayer(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LinearLayer,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim,output_dim),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self,x):
        return self.model(x)

@BLOCKS.registry("conv")
class ConvLayer(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 kernel_size,
                 stride,
                 padding,
                 batch_norm:bool=True):
        super().__init__()

        self.model = [
            nn.Conv2d(in_channels=input_dim,out_channels=output_dim,kernel_size=kernel_size,stride=stride,padding=padding,bias=False)
            ]
        #add batch norm
        if batch_norm:
            self.model.append(nn.BatchNorm2d(output_dim))
        #add activation to each layer except the last
        self.model.append(nn.LeakyReLU(0.2,inplace=True))
        #combine to sequential object 
        self.model = nn.Sequential(*self.model)
        
    def forward(self,x):
        return self.model(x)


@BLOCKS.registry("deconv")
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




@BLOCKS.registry("resnet_up")
class ResNETLayerUp(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 batch_norm:bool=True):
        super().__init__()

        layers = []
        if batch_norm:
            layers.append(nn.BatchNorm2d(in_channels))
        layers.extend([nn.ReLU(),
                              nn.Upsample(scale_factor=2,mode="nearest"),])

        layers.append(nn.Conv2d(in_channels,out_channels,3,stride=1,padding=1))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.extend([nn.ReLU(),
                                  nn.Conv2d(out_channels,out_channels,3,stride=1, padding=1)])

        self.model = nn.Sequential(*layers)

        self.skip = nn.Conv2d(in_channels,out_channels,1)

    def forward(self,x):
        return F.interpolate(self.skip(x),scale_factor=2,mode="nearest") + self.model(x)

@BLOCKS.registry("resnet_down")
class ResNETLayerDown(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 batch_norm:bool=True):
        super().__init__()
        layers = []
        if batch_norm:
            layers.append(nn.BatchNorm2d(in_channels))
        layers.append(nn.Conv2d(in_channels,out_channels,3,stride=1,padding=1))
        layers.append(nn.LeakyReLU(0.2))
        
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Conv2d(out_channels,out_channels,3,stride=1, padding=1))
        layers.append(nn.AvgPool2d(2))
        self.model = nn.Sequential(*layers)

        self.skip = nn.Conv2d(in_channels,out_channels,1,stride=2)

    def forward(self,x):
        return self.skip(x) + self.model(x)


@BLOCKS.registry("minibatch_discrimination")
class MiniBatchDiscrimination(nn.Module):
    def __init__(self,
                 in_features,
                 num_kernels,
                 kernel_dim):
        super().__init__()
        self.num_kernels = num_kernels
        self.kernel_dim = kernel_dim
        self.in_features = in_features
        # self.T = nn.Linear(in_features,num_kernels * kernel_dim,bias=False)
        self.T = nn.Parameter(torch.randn(in_features,num_kernels*kernel_dim))

    def forward(self,x):
        #multiply with learnarble params
        # M = self.T(x)
        M = x @ self.T
        #reshape to bs,nk,kd
        M = M.view(x.size(0),self.num_kernels,self.kernel_dim)
        # pairwise distances
        diff = M.unsqueeze(0) - M.unsqueeze(1)
        #sum the differences
        distance = torch.abs(diff).sum(dim=3)
        # negative distance
        similarity = torch.exp(-distance)
        # aggregate over other samples
        o = similarity.sum(dim=1)
        x = torch.cat([x, o], dim=1)
        return x

@BLOCKS.registry("self_attention")
class SelfAttention(nn.Module):
    def __init__(self,in_features):
        super().__init__()
        #scaling factor
        self.k = 2
        self.f = nn.Conv2d(in_features,in_features//self.k,1,bias=False)
        self.g = nn.Conv2d(in_features,in_features//self.k,1,bias=False)

        self.h = nn.Conv2d(in_features,in_features,1)
        self.v = nn.Conv2d(in_features,in_features,1)
        #learnable scalar factor
        self.factor = nn.Parameter(torch.tensor(0.0))
        

    def forward(self,x):
        # encode shape
        B,C,W,H = x.shape
        # spatial shape CxN
        N = W * H
        #feature spaces
        f = self.f(x).view(B,C//self.k,N) #batch,channels,HxW
        g = self.g(x).view(B,C//self.k,N)
        #attention scores
        s = f.transpose(1,2) @ g #Q^T @ K
        #softmax for probabilities
        scores = F.softmax(s,dim=1) #over N
        # sum as input of self.v
        in_v= scores @ self.h(x).view(B,C,N).transpose(1,2)
        #reshape in_v
        in_v = in_v.transpose(1, 2).view(B, C, H, W)
        #out with in_v as argument of self.v
        out = self.v(in_v)
        # entire output 
        y = self.factor * out + x
        return y

