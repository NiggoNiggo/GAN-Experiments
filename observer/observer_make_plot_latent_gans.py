from .observer import Observer
import os
from testing.create_image import ConvGANImageSampler

class PlotLatentGANsObserver(Observer):

    def __init__(self, num_images=64):
        self.num_images = num_images

    def update(self, info):

        epoch = info["epoch"]
        trainer = info["trainer"]
        self.plotter = ConvGANImageSampler(trainer.gen,trainer.latent_dim)
        imgs = self.plotter.sample_images(self.num_images)
        self.plotter.plot_images_grid(imgs, num_img=self.num_images, nrow=8, normalize=True,filename=os.path.join(trainer.save_path,trainer.filename,"plots",f"epoch_{epoch}.png"))

        

