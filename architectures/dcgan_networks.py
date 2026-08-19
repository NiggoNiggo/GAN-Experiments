from .layers import ConvLayer, ConvTransposeLayer, ResNETLayerUp, ResNETLayerDown, MiniBatchDiscrimination
from core.registries import GENERATORS, DISCRIMINATORS
from torch import nn
import math
import torch
from regularisations import spectral_normalisation


@GENERATORS.registry("dcgan")
class DCGANGenerator(nn.Module):
    def __init__(self,
                 out_shape:int, 
                out_channels:int,
                latent_dim: int,
                block_type:str,
                activation:str="Tanh"):
        super().__init__()
        #num of layers to obtain 1x1 at the end with a given output shape
        #this only works for number 2^n
        num_layers = int(math.log2(out_shape)) - 1  
        start_channels = 2 ** (num_layers + 5)  

        

        #creates list to save input dims and output dims respectively 
        in_dims = [start_channels // (2**i) for i in range(num_layers-1)]
        out_dims = in_dims[1:] + [out_channels]

        #linear mapping network
        self.linear_mapping = nn.Linear(latent_dim,in_dims[0]*4*4)

        self.model = []
        for k in range(len(in_dims)):
            if block_type == "deconv":
                layer = ConvTransposeLayer(
                    input_dim=in_dims[k],
                    output_dim=out_dims[k],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    last_layer=(k == len(in_dims)-1)
                )

                #resnet down and upsample layer
                
            if block_type == "resnet":
                if k == len(in_dims)-1:
                    layer = nn.Sequential(
                        ResNETLayerUp(in_dims[k], in_dims[k]),
                        nn.Conv2d(in_dims[k], out_dims[k], kernel_size=3, padding=1)
                    )
                else:
                    layer = ResNETLayerUp(in_dims[k], out_dims[k])
                
            self.model.append(layer)

        self.model.append(getattr(nn,activation)())
        self.model = nn.Sequential(*self.model)
        

    def forward(self, x):
        bs = x.size(0)
        x = x.squeeze(-1).squeeze(-1)
        x = self.linear_mapping(x)
        x = x.view(bs,-1,4,4)

        return self.model(x)

@DISCRIMINATORS.registry("dcgan")
class DCGANDiscriminator(nn.Module):
    def __init__(self,
                 out_shape:int,
                 in_channels:int,
                 block_type:str,
                 activation:str,
                 spectral_norm:bool,
                 minibatch_discrimination:bool):
        super().__init__()
        self.model = []
        self.minibatch_discrimination = minibatch_discrimination
        #computes the amount of layers to append to the desired out shape
        num_layers = int(math.log2(out_shape)) -1
        #create the in dimensions with start 64 and increase the power of 2
        in_dims = [in_channels] + [2**(6+n) for n in range(num_layers-1)]
        out_dims = [2**(6+n) for n in range(num_layers-1)] + [1]
        
        #assert that in and out dims have the same amout of values
        assert len(in_dims) == len(out_dims)

        for k in range(num_layers):
            is_last = (k == num_layers - 1)
            #if is the last layer and minibatch discrimination is inacitve
            if is_last and not self.minibatch_discrimination:
                if block_type == "deconv":
                    layer = ConvLayer(
                        input_dim=in_dims[k],
                        output_dim=out_dims[k],
                        kernel_size=4,
                        stride=1 if is_last else 2,
                        padding=0 if is_last else 1,
                        batch_norm=False #(k > 0)
                    )
                elif block_type == "resnet":
                    if is_last:
                        # last block: 4x4 -> 1x1
                        layer = nn.Conv2d(
                            in_dims[k],
                            out_dims[k],
                            kernel_size=4,
                            stride=1,
                            padding=0
                        )
                    else:
                        layer = ResNETLayerDown(in_dims[k],out_dims[k],batch_norm=False)
                self.model.append(layer)
            #normal case is not the last layer any intermediate layer
            elif not is_last:
                if block_type == "deconv":
                    layer = ConvLayer(
                        input_dim=in_dims[k],
                        output_dim=out_dims[k],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        batch_norm=False
                    )
                elif block_type == "resnet":
                    layer = ResNETLayerDown(
                        in_dims[k],
                        out_dims[k],
                        batch_norm=False
                    )
                self.model.append(layer)
            
            elif is_last and self.minibatch_discrimination:
                feature_channels = in_dims[k]
                feature_size = 4
                in_features = feature_channels * feature_size * feature_size

                mbd = MiniBatchDiscrimination(in_features=in_features,
                                            num_kernels=100,
                                            kernel_dim=5)
                self.model.append(nn.Flatten())
                self.model.append(mbd)
                self.model.append(nn.Linear(in_features+100,1))


        if activation.lower() != "none":
            activation = getattr(torch.nn, activation)()
            self.model.append(activation)

        self.model = nn.Sequential(*self.model)
        if spectral_norm:
            spectral_normalisation.apply_spectral_normalization(self.model)

        self.feature_layer = len(self.model) // 2
        
    def forward(self, x, return_features=False):
        features = None
        for i, layer in enumerate(self.model):
            x = layer(x)
            #feature layer for feature matching
            if return_features and i == self.feature_layer:
                features = x
        if return_features:
            return x.view(-1), features
        return x.view(-1)





if __name__ == "__main__":
    import torch
    # Testing the Generator
    print(f"Test Generator: create Noise and propagate it in the Generator, if worked output shapes of z and Fake are shown....")
    channels = 3
    z = torch.randn((64,100,1,1))
    out_shape = 32
    gen = DCGANGenerator(out_shape=out_shape,out_channels=channels,latent_dim=100,block_type="resnet")
    fake = gen(z)
    print(f"Z: {z.shape}, Fake: {fake.shape}")
    # Testing the Discriminator
    print(f"Test Discriminator: create Noise with batchsize (because if not batchsize is propagated the network crashes due to batchnorm) and propagate it in the Discriminator, if worked output shapes of noise and discriminator output are shown....")
    
    # in_dims = [channels,64,128,256]
    # out_dims = [64,128,256,1]
    disc = DCGANDiscriminator(out_shape=out_shape,in_channels=channels,block_type="resnet",activation="Sigmoid")
    print(disc)
    noise = torch.randn((64,channels,out_shape,out_shape))
    print("Noise ", noise.shape)
    
    disc_out = disc(noise)
    
    print(f"Fake shape: {fake.shape}, Disc Out shape: {disc_out.shape}")

    print(gen)


  