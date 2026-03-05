from .layers import ConvLayer, ConvTransposeLayer
from torch import nn
import math


class DCGANGenerator(nn.Module):
    def __init__(self,out_shape,in_dims,out_dims):
        super().__init__()
        #computes the amount of layers to append to the desired out shape
        num_layers = int(math.log2(out_shape) -1)
        
        #check that lengths of input and output channels match with the num layes calculated here
        assert isinstance(num_layers, int)
        assert len(in_dims) == num_layers
        assert len(out_dims) == num_layers
        #raise faults if the shape of the inn_dims and out_dims do not match with the bum_layers calculated above (normally, -2 but duo to I assume the first layer, that is not for scaling to be created within the loop so we need one more iteration to create this layer)
        
        self.model = []
        #builds the actual model. This works dynamically for each shape, that is 2^n % 0.
        for k in range(num_layers):
            layer = ConvTransposeLayer(input_dim=in_dims[k],output_dim=out_dims[k],kernel_size=4,stride=2,padding=0 if k == 0 else 1,last_layer=True if k == num_layers-1 else False)
            self.model.append(layer)
                
        #config model to a nn.Sequential model        
        self.model = nn.Sequential(*self.model)
    
    
    def forward(self,x):
        return self.model(x)


class DCGANDiscriminator(nn.Module):
    def __init__(self,out_shape,in_dims,out_dims):
        super().__init__()
        self.model = []
        #computes the amount of layers to append to the desired out shape
        num_layers = int(math.log2(out_shape) -1)
        for k in range(num_layers):
            layer = ConvLayer(input_dim=in_dims[k],output_dim=out_dims[k],kernel_size=4,stride=2,padding=0 if k == 0 else 1,
            batch_norm=False if k == 0 and k == num_layers-1 else True,
            last_layer=True if k == num_layers-1 else False)
            self.model.append(layer)
        
        # self.model.append(nn.Sigmoid())
        self.model = nn.Sequential(*self.model)
        
        
    def forward(self,x):
        return self.model(x)


if __name__ == "__main__":
    import torch
    # Testing the Generator
    print(f"Test Generator: create Noise and propagate it in the Generator, if worked output shapes of z and Fake are shown....")
    z = torch.randn((1,100,1,1))
    print(z.shape)
    out_shape = 32
    in_dims = [100,64,32,16]
    out_dims = [64,32,16,1]
    num_layers = math.log2(out_shape) -2
    print(num_layers)
    gen = DCGANGenerator(out_shape=out_shape,in_dims=in_dims,out_dims=out_dims)
    fake = gen(z)
    print(f"Z: {z.shape}, Fake: {fake.shape}")
    
    # Testing the Discriminator
    print(print(f"Test Discriminator: create Noise with batchsize (because if not batchsize is propagated the network crashes due to batchnorm) and propagate it in the Discriminator, if worked output shapes of noise and discriminator output are shown...."))
    
    in_dims = [1,64,128,256]
    out_dims = [64,128,256,1]
    disc = DCGANDiscriminator(out_shape=out_shape,in_dims=in_dims,out_dims=out_dims)
    
    noise = torch.randn((64,1,32,32))
    
    disc_out = disc(noise)
    
    print(f"Fake shape: {fake.shape}, Disc Out shape: {disc_out.shape}")
    print(gen)
    print(disc)