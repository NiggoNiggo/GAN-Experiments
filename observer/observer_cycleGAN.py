import os
import torch

class CycleGANImageObserver:

    def __init__(self, interval=1, num_images=4):
        self.interval = interval
        self.num_images = num_images

    def update(self, info):

        epoch = info["epoch"]
        trainer = info["trainer"]

        if epoch % self.interval != 0:
            return

        batch = next(iter(trainer.data_loader))

        trainer.G_AB.eval()
        trainer.G_BA.eval()

        with torch.no_grad():

            real_A = batch["A"].to(trainer.device)[:self.num_images]
            real_B = batch["B"].to(trainer.device)[:self.num_images]

            fake_B = trainer.G_AB(real_A)
            fake_A = trainer.G_BA(real_B)

        def denorm(x):
            return (x + 1) / 2

        real_A = denorm(real_A)
        real_B = denorm(real_B)
        fake_B = denorm(fake_B)
        fake_A = denorm(fake_A)

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(self.num_images, 4, figsize=(10, 2*self.num_images))

        for i in range(self.num_images):

            imgs = [
                real_A[i],
                fake_B[i],
                real_B[i],
                fake_A[i]
            ]

            titles = [
                "real_A",
                "A → B",
                "real_B",
                "B → A"
            ]

            for j in range(4):

                img = imgs[j].cpu().permute(1, 2, 0)

                axes[i, j].imshow(img)
                axes[i, j].set_title(titles[j])
                axes[i, j].axis("off")

        plt.tight_layout()

        save_dir = f"{trainer.save_path}/{trainer.filename}/plots"
        os.makedirs(save_dir, exist_ok=True)

        plt.savefig(f"{save_dir}/epoch_{epoch}.png")
        plt.close()

        trainer.G_AB.train()
        trainer.G_BA.train()