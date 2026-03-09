from training.vanilla_trainer import VanillaGANTrainer
from training.dcgan_trainer import DCGANTrainer
from training.cgan_trainier import ConditionalDCGANTrainer
from architectures.linear_networks import LinearDiscriminator, LinearGenerator
from architectures.dcgan_networks import DCGANDiscriminator, DCGANGenerator
from architectures.cgan_networks import ConditionalDCGANDiscriminator, ConditonalDCGANGenerator
from param_configs.optimizer_factory import AdamStrategy
from loss_functions.vanilla_loss import VanillaGANLoss
from data.mnist import mnist_train, mnist_test
from data.celebA import celeb_dataset
from data.wrappers import DataWrapper
from testing.create_image import LinearGANImageSampler, ConditionalGANImageSampler, ConvGANImageSampler

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
# out_shape = 16
# channels = 3
# latent_dim = 100

# training = DCGANTrainer(
#     gen=DCGANGenerator(out_shape=out_shape,out_channels=channels,latent_dim=latent_dim),
#     disc=DCGANDiscriminator(out_shape=out_shape,in_channels=channels),
#     data_loader=torch.utils.data.DataLoader(DataWrapper(celeb_dataset,has_labels=False),batch_size=128,shuffle=True),
#     loss_fn=VanillaGANLoss(),
#     optim_gen_strat=AdamStrategy(lr=0.0002, betas=(0.5, 0.999)),
#     optim_disc_strat=AdamStrategy(lr=0.0002, betas=(0.5, 0.999)),
#     latent_dim=100)

# gen, disc = training.train(100)
# sample = SampleImages(gen,100)
# sample.sample_images(64)


#Conditional Trainer
out_shape = 32
channels = 1
latent_dim = 100
num_classes = 10

training = ConditionalDCGANTrainer(
    gen=ConditonalDCGANGenerator(out_shape=out_shape,out_channels=channels,latent_dim=latent_dim,num_classes=num_classes),
    disc=ConditionalDCGANDiscriminator(out_shape=out_shape,in_channels=channels,num_classes=10),
    data_loader=torch.utils.data.DataLoader(DataWrapper(mnist_train,has_labels=True),batch_size=128,shuffle=True),
    loss_fn=VanillaGANLoss(),
    optim_gen_strat=AdamStrategy(lr=0.0002, betas=(0.5, 0.999)),
    optim_disc_strat=AdamStrategy(lr=0.0002, betas=(0.5, 0.999)),
    latent_dim=100)

gen, disc = training.train(10)

images = ConditionalGANImageSampler(gen,latent_dim,num_classes=num_classes)
imgs = images.sample_images(6*10)
images.plot_images_grid(imgs,60,nrow=6)