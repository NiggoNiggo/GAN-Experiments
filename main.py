import os
from pathlib import Path
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
from loss_functions.lsgan_loss import LSLoss
from loss_functions.hinge_loss import HingeLoss
# from data.mnist import mnist_train, mnist_test

from data.celebA import CelebADataset
from data.wrappers import DataWrapper
from data.cycle_dataset import CycleDataset
from testing.create_image import LinearGANImageSampler, ConditionalGANImageSampler, ConvGANImageSampler
from observer.observer_save import ModelSaver
from observer.observer_plot_values import PlotObserver
from observer.observer_make_plot_latent_gans import PlotLatentGANsObserver
from observer.observer_evaluation import EvalObserver
# from plotting.loss_plotting import Plotting
import multiprocessing

if __name__ == "__main__":
    #make training more efficient
    multiprocessing.set_start_method('spawn', force=True)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(8) 
 
    # DCGAN Trainer
    out_shape = 256
    channels = 3
    latent_dim = 100

    
    transform = transforms.Compose([
    transforms.Resize((out_shape, out_shape)),      
    transforms.ToTensor(),             
    transforms.Normalize((0.5,), (0.5,), (0.5,))  
    ])


    # data_path = Path("/mnt/data2/datasets/celebA/img_align_celeba")
    # data_path = Path("/mnt/data2/datasets/ImageNET_half")
    data_path = Path("/mnt/data2/datasets/lsun/train")
    save_path = Path("/mnt/data2/gan_results")
    dataset = CelebADataset(data_path,transform)
    filename = "lsgan_lsun_256x256"

    training = DCGANTrainer(
        gen=DCGANGenerator(out_shape=out_shape,out_channels=channels,latent_dim=latent_dim),
        disc=DCGANDiscriminator(out_shape=out_shape,in_channels=channels),
        data_loader=torch.utils.data.DataLoader(DataWrapper(dataset,has_labels=False),batch_size=256,shuffle=True,pin_memory=True,num_workers=10,persistent_workers=True),
        loss_fn=LSLoss(),
        optim_gen_strat=AdamStrategy(lr=0.0002, betas=(0.5, 0.999)),
        optim_disc_strat=AdamStrategy(lr=0.0002, betas=(0.5, 0.999)),
        latent_dim=100,
        save_path=save_path,
        filename=filename)
    training.attach(ModelSaver(save_path=save_path / filename))
    training.attach(EvalObserver())
    training.attach(PlotObserver(path=os.path.join(save_path, filename),filename="values.csv"))
    training.attach(PlotLatentGANsObserver(num_images=64))

    #resnet gan einbauen





    # plotter = Plotting(save_path, filename)
    # plot_observer = PlotObserver(plotter)
    # training.attach(plot_observer)


    training.train(150)
    # sample = ConvGANImageSampler(gen,100)
    # sample.sample_images(64)



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

