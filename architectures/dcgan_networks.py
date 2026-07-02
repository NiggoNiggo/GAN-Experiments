from .layers import ConvLayer, ConvTransposeLayer
from torch import nn
import math


class DCGANGenerator(nn.Module):
    def __init__(self, out_shape, out_channels: int, latent_dim: int):
        super().__init__()

        # Anzahl der Layer bis out_shape von 1x1
        num_layers = int(math.log2(out_shape)) - 1  # z.B. 32 -> 5 Layer

        start_channels = 2 ** (num_layers + 5)  # z.B. 1024 für 32x32

        # Listen für in/out Channels
        in_dims = [latent_dim] + [start_channels // (2**i) for i in range(num_layers-1)]
        out_dims = [start_channels // (2**i) for i in range(num_layers-1)] + [out_channels]

        print("Generator in/out dims:", in_dims, out_dims)

        self.model = []
        for k in range(len(in_dims)):
            layer = ConvTransposeLayer(
                input_dim=in_dims[k],
                output_dim=out_dims[k],
                kernel_size=4,
                stride=2,
                padding=0 if k == 0 else 1,
                last_layer=(k == len(in_dims)-1)
            )
            self.model.append(layer)

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class DCGANDiscriminator(nn.Module):
    def __init__(self,out_shape,in_channels,net_type="lsgan"):
        super().__init__()
        self.model = []
        #computes the amount of layers to append to the desired out shape
        num_layers = int(math.log2(out_shape)) -1
        #create the in dimensions with start 64 and increase the power of 2
        in_dims = [in_channels] + [2**(6+n) for n in range(num_layers-1)]
        out_dims = [2**(6+n) for n in range(num_layers-1)] + [1]
        print("out dims: ", out_dims)
        print("in dims: ", in_dims)
        print(num_layers)
        print(len(in_dims),len(out_dims))
        #assert that in and out dims have the same amout of values
        assert len(in_dims) == len(out_dims)
        #assert that in_channels have equally values containing as num layers' length
        # assert len(in_dims) == num_layers
        # #assert that the length of out dims matches num layers
        # assert len(out_dims) == num_layers
        

        
        for k in range(num_layers):
            output_dim = out_dims[k] if k < num_layers-1 else in_channels
            last_layer = (k == num_layers-1)
            layer = ConvLayer(
                input_dim=in_dims[k],
                output_dim=output_dim,
                kernel_size=4,
                stride=1 if k == num_layers-1 else 2,
                padding=0 if k == num_layers-1 else 1,
                last_layer=last_layer,
                batch_norm=(k>0)
            )

            self.model.append(layer)
        
        if last_layer:
            if net_type in ["lsgan","wgan"]:
                self.model[-1].model = self.model[-1].model[:-1]
                # self.model.append(nn.Flatten())
                # self.model.append(nn.Linear(in_dims[-1],3))
        
        self.model = nn.Sequential(*self.model)
        
        
    def forward(self,x):
        return self.model(x)


if __name__ == "__main__":
    import torch
    # Testing the Generator
    print(f"Test Generator: create Noise and propagate it in the Generator, if worked output shapes of z and Fake are shown....")
    
    channels = 3
    z = torch.randn((64,100,1,1))
    out_shape = 64
    gen = DCGANGenerator(out_shape=out_shape,out_channels=channels,latent_dim=100)
    fake = gen(z)
    print(f"Z: {z.shape}, Fake: {fake.shape}")
    # Testing the Discriminator
    print(f"Test Discriminator: create Noise with batchsize (because if not batchsize is propagated the network crashes due to batchnorm) and propagate it in the Discriminator, if worked output shapes of noise and discriminator output are shown....")
    
    # in_dims = [channels,64,128,256]
    # out_dims = [64,128,256,1]
    disc = DCGANDiscriminator(out_shape=out_shape,in_channels=channels)
    noise = torch.randn((64,channels,out_shape,out_shape))
    print("Noise ", noise.shape)
    
    disc_out = disc(noise)
    
    print(f"Fake shape: {fake.shape}, Disc Out shape: {disc_out.shape}")
  