from .observer import Observer
import os
from testing.create_image import ConvGANImageSampler
import torch
class PlotLatentGANsObserver(Observer):

    def __init__(self, num_images=64):
        self.num_images = num_images

    def update(self, info):

        iterations= info["num_iterations"]
        trainer = info["trainer"]
        self.plotter = ConvGANImageSampler(trainer.gen,trainer.latent_dim)
        imgs = self.plotter.sample_images(self.num_images)
        self.plotter.plot_images_grid(imgs, num_img=self.num_images, nrow=8,filename=os.path.join(trainer.save_path,trainer.filename,"fake",f"num_iterations_{iterations//1000}k.png"))

        

