import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path



#should get his data concurrently in the training for regulary plotting and should be able for plotting in the end and outside of the training 
class Plotting:
    def __init__(self,
                 path:str,
                 filename:str):
        self.path = path
        self.filename = filename
        self.values = self.init_plotter()
    
    def init_plotter(self):
        #read entire csv file, containing all informations 
        values = pd.read_csv(Path(self.path) / self.filename / "values_csv" / "values_csv.csv",index_col=False)
        return values
    
    def update_values(self):
        #read the updated csv file
        values = self.init_plotter()
        #only if both dfs aren't equal return the new loaded df, otherwise pass
        if not self.values.equals(values):
            return values
    
    def plot_losses(self, save: bool, show: bool=True):
        loss_g = self.values["loss_g"]
        loss_d = self.values["loss_d"]
        x = self.values["epoch"]

        fig, ax = plt.subplots()

        ax.plot(x, loss_d, label="Discriminator Loss")
        ax.plot(x, loss_g, label="Generator Loss")

        ax.legend()
        ax.grid()
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")

        if save:
            self.save_plot(fig,"Loss_plot.png")

        if show:
            plt.show()

        plt.close(fig)
            
            
            
            
    
    def plot_metrics(self):
        pass
    
    def save_plot(self,
                  fig,
                  name : str):
        #here now enter the saving algorithm
        filename = Path(self.path) / self.filename / "plots" / name
        fig.savefig(filename)
        




class PlottingCycleGAN(Plotting):
    pass