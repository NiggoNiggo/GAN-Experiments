import os
from pathlib import Path
import matplotlib

from observer.observer_make_plot_latent_gans import PlotLatentGANsObserver
matplotlib.use('Agg')

import torch
from torchvision import transforms

from training.vanilla_trainer import VanillaGANTrainer
from training.dcgan_trainer import DCGANTrainer
from training.cgan_trainier import ConditionalDCGANTrainer
from training.CycleTrainer import CycleGANTrainer
from architectures.linear_networks import LinearDiscriminator, LinearGenerator
from architectures.dcgan_networks import DCGANDiscriminator, DCGANGenerator
from architectures.cgan_networks import ConditionalDCGANDiscriminator, ConditonalDCGANGenerator
from architectures.cycle_networks import CycleDiscriminator, CycleGenerator
from param_configs.optimizer_factory import AdamStrategy,CycleStrategy
from loss_functions.vanilla_loss import VanillaGANLoss
from data.mnist import mnist_train, mnist_test
from data.celebA import CelebADataset
from data.wrappers import DataWrapper
from data.cycle_dataset import CycleDataset
from testing.create_image import LinearGANImageSampler, ConditionalGANImageSampler, ConvGANImageSampler
from observer.observer_save import ModelSaver
from observer.observer_plot_values import PlotObserver
from observer.observer_cycleGAN import CycleGANImageObserver
from observer.observer_evaluation import EvalObserver
from plotting.loss_plotting import Plotting

if __name__ == "__main__":
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


    # DCGAN Trainer
    out_shape = 64
    channels = 3
    latent_dim = 100



    
    transform = transforms.Compose([
    transforms.Resize((out_shape, out_shape)),      
    transforms.ToTensor(),             
    transforms.Normalize((0.5,), (0.5,), (0.5,))  
    ])


    path = Path("/mnt/data2/datasets/monet/trainA")
    save_path = Path("/mnt/data2/gan_results")
    celeb_dataset = CelebADataset(path,transform)
    filename = "dc_test2"

    training = DCGANTrainer(
        gen=DCGANGenerator(out_shape=out_shape,out_channels=channels,latent_dim=latent_dim),
        disc=DCGANDiscriminator(out_shape=out_shape,in_channels=channels),
        data_loader=torch.utils.data.DataLoader(DataWrapper(celeb_dataset,has_labels=False),batch_size=64,shuffle=True,pin_memory=True,num_workers=8),
        loss_fn=VanillaGANLoss(),
        optim_gen_strat=AdamStrategy(lr=0.0002, betas=(0.5, 0.999)),
        optim_disc_strat=AdamStrategy(lr=0.0002, betas=(0.5, 0.999)),
        latent_dim=100,
        save_path=save_path,
        filename=filename)
    training.attach(ModelSaver(save_path=save_path / filename))
    training.attach(EvalObserver())
    training.attach(PlotObserver(path=os.path.join(save_path, filename),filename="values.csv"))
    training.attach(PlotLatentGANsObserver(num_images=64))

    # plotter = Plotting(path, filename)
    # plot_observer = PlotObserver(plotter)
    # training.attach(plot_observer)


    gen, disc = training.train(100)
    sample = ConvGANImageSampler(gen,100)
    sample.sample_images(64)



    # #Conditional Trainer
    # out_shape = 32
    # channels = 1
    # latent_dim = 100
    # num_classes = 10
    # save_path = "D:\DeepLearning_Results"
    # filename = "test_cgan"
    # csv_values = "values_csv"

    # full_test_path = save_path / filename

    # training = ConditionalDCGANTrainer(
    #     save_path=save_path,
    #     filename=filename,
    #     gen=ConditonalDCGANGenerator(out_shape=out_shape,out_channels=channels,latent_dim=latent_dim,num_classes=num_classes),
    #     disc=ConditionalDCGANDiscriminator(out_shape=out_shape,in_channels=channels,num_classes=10),
    #     data_loader=torch.utils.data.DataLoader(DataWrapper(mnist_train,has_labels=True),batch_size=128,shuffle=True),
    #     loss_fn=VanillaGANLoss(),
    #     optim_gen_strat=AdamStrategy(lr=0.0002, betas=(0.5, 0.999)),
    #     optim_disc_strat=AdamStrategy(lr=0.0002, betas=(0.5, 0.999)),
    #     latent_dim=100,
        
    # )

    # training.attach(ModelSaver(save_path=full_test_path))
    # training.attach(PlotObserver(full_test_path,filename=csv_values))
    # # training.attach(SampleLogger(sample_path="./samples"))


    # gen, disc = training.train(2)
    # training.attach(PlotLatentGANsObserver(num_images=64))

    # make_plots = Plotting(path=save_path,filename=filename)
    # make_plots.plot_losses(show=True,save=True)

    # images = ConditionalGANImageSampler(gen,latent_dim,num_classes=num_classes)
    # imgs = images.sample_images(6*10)
    # images.plot_images_grid(imgs,60,nrow=6)




    #-----CycleGAN
    # latent_dim = 100
    # save_path = Path(r"/mnt/data2/gan_results")
    # filename = "monet_photo_cycleGAN"
    # csv_values = "values_csv"
    # path_A = Path("/mnt/data2/datasets/monet/trainA")
    # path_B = Path("/mnt/data2/datasets/photo/trainB")

    # full_test_path = save_path / filename


    # data_loader = torch.utils.data.DataLoader(
    #     CycleDataset(path_A=path_A, path_B=path_B, transform=None),
    #     batch_size=1,
    #     shuffle=True,
    #     pin_memory=True,
    #     num_workers=8,
    #     drop_last=True)

    # print(len(data_loader))

    # training = CycleGANTrainer(
    #     G_AB=CycleGenerator(),
    #     G_BA=CycleGenerator(),
    #     D_A=CycleDiscriminator(),
    #     D_B=CycleDiscriminator(),
    #     data_loader=data_loader,
    #     loss_fn=VanillaGANLoss(),
    #     optim_strat=CycleStrategy(lr=0.0002, betas=(0.5, 0.999)),
    #     latent_dim=100,
    #     save_path=save_path,
    #     filename=filename
    # )    
    # training.attach(CycleGANImageObserver())
    # training.attach(ModelSaver(full_test_path))
        
    # training.train(50)
    
    #loss plot observer adaption to cyclegan
    
    #what do i need next?
    
    
    #experimente mit der latent dim machne mal
    #after SRGAN and dynamic cyclegan size
    #stackGAN auch implementieren



    #rgbs anpassen