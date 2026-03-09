from .layers import ConvLayer, ConvTransposeLayer
from torch import nn
import torch
import math


class ConditonalDCGANGenerator(nn.Module):
    def __init__(self, 
                out_shape: int,
                out_channels: int,
                latent_dim: int,
                num_classes: int):
        super().__init__()
        #num classes 
        self.label_emb = nn.Embedding(num_classes, num_classes)
        # Anzahl der Layer bis out_shape von 1x1
        num_layers = int(math.log2(out_shape)) - 1  # z.B. 32 -> 5 Layer

        start_channels = 2 ** (num_layers + 5)  # z.B. 1024 für 32x32

        # Listen für in/out Channels
        in_dims = [latent_dim] + [start_channels // (2**i) for i in range(num_layers-1)]
        out_dims = [start_channels // (2**i) for i in range(num_layers-1)] + [out_channels]


        self.model = []
        for k in range(len(in_dims)):
            layer = ConvTransposeLayer(
                input_dim=in_dims[k] + num_classes if k == 0 else in_dims[k],
                output_dim=out_dims[k],
                kernel_size=4,
                stride=2,
                padding=0 if k == 0 else 1,
                last_layer=(k == len(in_dims)-1)
            )
            self.model.append(layer)

        self.model = nn.Sequential(*self.model)

    def forward(self, x,labels):
        labels = self.label_emb(labels)
        labels = labels.unsqueeze(2).unsqueeze(3)
        x = torch.cat([x,labels],dim=1)
        return self.model(x)


class ConditionalDCGANDiscriminator(nn.Module):
    def __init__(self,
                 out_shape: int,
                 in_channels: int,
                 num_classes: int):
        super().__init__()
        self.out_shape = out_shape
        #embedding for labels
        self.label_emb = torch.nn.Embedding(num_classes,num_classes)
        self.model = []
        #computes the amount of layers to append to the desired out shape
        num_layers = int(math.log2(out_shape))
        #create the in dimensions with start 64 and increase the power of 2
        in_dims = [in_channels] + [2**(6+n) for n in range(num_layers-1)]
        out_dims = [2**(6+n) for n in range(num_layers-1)] + [1]
        #assert that in and out dims have the same amout of values
        assert len(in_dims) == len(out_dims)
        #assert that in_channels have equally values containing as num layers' length
        # assert len(in_dims) == num_layers
        # #assert that the length of out dims matches num layers
        # assert len(out_dims) == num_layers
        

        
        for k in range(num_layers):
            output_dim = out_dims[k] if k < num_layers-1 else in_channels
            layer = ConvLayer(
                input_dim=in_dims[k] + num_classes if k == 0 else in_dims[k],
                output_dim=output_dim,
                kernel_size=2 if k == num_layers-1 else 4,
                stride=2,
                padding=0 if k == num_layers-1 else 1,
                last_layer=(k == num_layers-1)
            )

            self.model.append(layer)
        
        # self.model.append(nn.Sigmoid())
        self.model = nn.Sequential(*self.model)
        
        
    def forward(self,x,labels):
        labels = self.label_emb(labels)
        labels = labels.unsqueeze(2).unsqueeze(3)
        labels = labels.repeat(1,1,self.out_shape,self.out_shape)
        x = torch.cat([x,labels],dim=1)
        return self.model(x)


if __name__ == "__main__":
    import torch
    # Testing the Generator
    print(f"Test Generator: create Noise and propagate it in the Generator, if worked output shapes of z and Fake are shown....")
    labels = 10
    channels = 1
    z = torch.randn((64,100,1,1))
    classes = torch.randint(0,labels,(64,))
    out_shape = 64
    gen = ConditonalDCGANGenerator(out_shape=out_shape,out_channels=channels,latent_dim=100,num_classes=labels)
    fake = gen(z,classes)
    print(f"Z: {z.shape}, Fake: {fake.shape}")
    # Testing the Discriminator
    print(f"Test Discriminator: create Noise with batchsize (because if not batchsize is propagated the network crashes due to batchnorm) and propagate it in the Discriminator, if worked output shapes of noise and discriminator output are shown....")
    

    disc = ConditionalDCGANDiscriminator(out_shape=out_shape,in_channels=channels,num_classes=labels)
    noise = torch.randn((64,channels,out_shape,out_shape))
    print("Noise ", noise.shape)
    
    disc_out = disc(noise,classes)
    
    print(f"Fake shape: {fake.shape}, Disc Out shape: {disc_out.shape}")
  