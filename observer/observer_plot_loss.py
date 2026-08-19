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

    def make_plot(self,
                  data:pd.DataFrame):
        #define x-axis vector
        t = data["num_iterations"]
        fig,ax = plt.subplots(ncols=2)
        #plot losses in one plot 
        ax[0].set_title("Losses Gen and disc")
        ax[0].set_ylabel("Loss")
        ax[0].set_xlabel("Iterations")
        ax[0].plot(t,data["loss_d"],label="Disc Loss")
        ax[0].plot(t,data["loss_g"],label="Gen Loss")
        #plot metrics in another loss
        ax[1].set_title("FID, IS and KIS")
        ax[1].set_ylabel("Metric")
        ax[1].set_xlabel("Iterations")
        ax[1].plot(t,data["FID"],label="FID")
        ax[1].plot(t,data["IS"],label="IS")
        ax[1].plot(t,data["KID"],label="KID")
        plt.legend()
        return fig


        