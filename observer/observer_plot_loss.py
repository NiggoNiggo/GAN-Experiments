import os
import pandas as pd
import matplotlib.pyplot as plt

from .observer import Observer


class PlotObserver(Observer):
    def __init__(self):
        super().__init__()


    def update(self,info):
        trainer = info["trainer"]
        #filename and path of the csv data 
        path_csv = os.path.join(trainer.project_path,"values_csv","values.csv")
        #read csv 
        data = pd.read_csv(path_csv,index_col=False)
        fig = self.make_plot(data)
        #save it directly in the project folder 
        save_path = os.path.join(trainer.project_path,"losses_plot.png")
        plt.savefig(save_path)
        plt.close()

    def make_plot(self, data: pd.DataFrame):
        # define x-axis vector
        t = data["num_iterations"]
        fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
        # Losses
        ax[0].set_title("Losses Gen and Disc")
        ax[0].set_ylabel("Loss")
        ax[0].set_xlabel("Iterations")
        ax[0].plot(t, data["loss_d"], label="Disc Loss")
        ax[0].plot(t, data["loss_g"], label="Gen Loss")
        ax[0].legend()
        # FID
        ax[1].set_title(f"FID min: {min(data["FID"])}")
        ax[1].set_xlabel("Iterations")
        ax[1].plot(t, data["FID"], label="FID")
        ax[1].legend()
        # IS
        ax[2].set_title("IS")
        ax[2].set_xlabel("Iterations")
        ax[2].plot(t, data["IS"], label=f"IS max: {max(data["IS"])}")
        ax[2].legend()
        # KID
        ax[3].set_title(f"KID min: {min(data["KID"])}")
        ax[3].set_xlabel("Iterations")
        ax[3].plot(t, data["KID"], label="KID")
        ax[3].legend()
        fig.tight_layout()

        return fig


        