import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class PlotLossValues:
    def __init__(self,
                 path):
        #actual path to the project folder including the filename of the project
        self.path = path
        #load the csv file that includes the losses and metrics
        self.all_data = self._load_data(path)

    def _load_data(self,path):
        #load csv values
        all_data = pd.read_csv(os.path.join(self.path,"values_csv"),index_col=False)
        return all_data

    def make_plot(self):
        #create the plot
        fix,ax = plt.subplots()
        #x axis for plotting the loss
        x = np.arange(len(self.all_data["loss_d"].values))
        #path where to save the plot
        save_path = os.path.join(self.path,"loss_plot")
        ax.plot(x,self.all_data["loss_d"].values,label="Loss D")
        ax.plot(x,self.all_data["loss_g"].values,label="Loss G")
        plt.grid()
        plt.legend(True)
        plt.savefig(save_path)


