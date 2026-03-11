from .trainer import GANTrainer
import torch
# from testing.create_image import ConditionalGANImageSampler

class ConditionalDCGANTrainer(GANTrainer):
    def __init__(self, gen, disc, data_loader, loss_fn, optim_gen_strat, optim_disc_strat, latent_dim,save_path,filename,device="cuda"):
        super().__init__(gen, disc, data_loader, loss_fn, optim_gen_strat, optim_disc_strat,latent_dim,save_path=save_path,filename=filename)
        self.latent_dim = latent_dim
        self.device = device
        self.verbose = False #default False
        # self.num_classes = torch.unique(self.data_loader.)
        self.gen.to(device)
        self.disc.to(device)
        #build optimizers
        self.optim_gen = optim_gen_strat.build_optim(self.gen)
        self.optim_disc = optim_disc_strat.build_optim(self.disc)

        # create the image plotter now that generator has been moved to the correct device
        # self.image_plotter = SampleImages(self.gen, self.latent_dim)

    def train_disc(self, real_data):
        batch_size, real, labels = self.ensure_correct_input(real_data)
        
        real_pred = self.disc(real,labels).view(-1)
        z = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
        fake_data = self.gen(z,labels).detach()
        # shape should already be (batch, output_dim)
        fake_pred = self.disc(fake_data,labels).view(-1)
        d_loss = self.loss_fn.disc_loss(real_pred, fake_pred)
        
        self.optim_disc.zero_grad()
        d_loss.backward()
        self.optim_disc.step()

        return d_loss.item()

    def train_gen(self, batch_size,labels):
        z = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
        fake_data = self.gen(z,labels)
        fake_pred = self.disc(fake_data,labels).view(-1)
        g_loss = self.loss_fn.gen_loss(fake_pred)
        self.optim_gen.zero_grad()
        g_loss.backward()
        self.optim_gen.step()

        return g_loss.item()


    def train_step(self, batch):
        d_loss = self.train_disc(batch)
        batch_size, real, labels = self.ensure_correct_input(batch)
        g_loss = self.train_gen(batch_size,labels)
        # images = ConditionalGANImageSampler(self.gen,self.latent_dim,num_classes=self.num_classes)
        # imgs = images.sample_images(6*10)
        # images.plot_images_grid(imgs,60,nrow=10)
        return d_loss, g_loss