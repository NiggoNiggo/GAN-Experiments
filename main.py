from training.vanilla_trainer import VanillaGANTrainer
from training.dcgan_trainer import DCGANTrainer
from architectures.linear_networks import LinearDiscriminator, LinearGenerator
from architectures.dcgan_networks import DCGANDiscriminator, DCGANGenerator
from param_configs.optimizer_factory import AdamStrategy
from loss_functions.vanilla_loss import VanillaGANLoss
from data.mnist import train_data, test_data
from data.wrappers import DataWrapper
from testing.create_image import SampleImages

import torch

#Vanilla Linear GAN Trainer:

# training = VanillaGANTrainer(
#     gen=LinearGenerator(input_dim=100, output_dim=28*28),
#     disc=LinearDiscriminator(input_dim=28*28),
#     data_loader=torch.utils.data.DataLoader(DataWrapper(train_data,has_labels=True),batch_size=64,shuffle=True),
#     loss_fn=VanillaGANLoss(),
#     optim_gen_strat=AdamStrategy(lr=0.0002, betas=(0.5, 0.999)),
#     optim_disc_strat=AdamStrategy(lr=0.0002, betas=(0.5, 0.999)),
#     latent_dim=100)

# gen, disc = training.train(100)


# sample = SampleNormalImages(gen,100)
# sample.sample_images(64)


#DCGAN Trainer
out_shape = 32
in_dims_gen = [100,256,128,64]
out_dims_gen = [256,128,64,1]
in_dims_disc = [1,64,128,256]
out_dims_disc = [64,128,256,1]

training = DCGANTrainer(
    gen=DCGANGenerator(out_shape=out_shape,out_dims=out_dims_gen,in_dims=in_dims_gen),
    disc=DCGANDiscriminator(out_shape=out_shape,in_dims=in_dims_disc,out_dims=out_dims_disc),
    data_loader=torch.utils.data.DataLoader(DataWrapper(train_data,has_labels=True),batch_size=128,shuffle=True),
    loss_fn=VanillaGANLoss(),
    optim_gen_strat=AdamStrategy(lr=0.0002, betas=(0.5, 0.999)),
    optim_disc_strat=AdamStrategy(lr=0.0002, betas=(0.5, 0.999)),
    latent_dim=100)

print(training.gen)
print(training.disc)
gen, disc = training.train(100)

    

# sample = SampleImages(gen,100)
# sample.sample_images(64)