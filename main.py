from training.vanilla_trainer import VanillaGANTrainer
from architectures.linear_networks import LinearDiscriminator, LinearGenerator
from param_configs.optimizer_factory import AdamStrategy
from loss_functions.vanilla_loss import VanillaGANLoss
from data.mnist import train_data, test_data
from data.wrappers import DataWrapper
from testing.create_image import SampleNormalImages

import torch


training = VanillaGANTrainer(
    gen=LinearGenerator(input_dim=100, output_dim=28*28),
    disc=LinearDiscriminator(input_dim=28*28),
    data_loader=torch.utils.data.DataLoader(DataWrapper(train_data,has_labels=True),batch_size=64,shuffle=True),
    loss_fn=VanillaGANLoss(),
    optim_gen_strat=AdamStrategy(lr=0.0002, betas=(0.5, 0.999)),
    optim_disc_strat=AdamStrategy(lr=0.0002, betas=(0.5, 0.999)),
    latent_dim=100)

gen, disc = training.train(100)

    

sample = SampleNormalImages(gen,100)
sample.sample_images(64)
