import os
import pandas as pd
from pathlib import Path

from .observer import Observer


# observer for loss values and plotting prepare it for metrics
#plotting class visulizer


class PlotObserver(Observer):
    def __init__(self,
                 path : str,
                 filename : str):
        #path goes until filename including
        self.path = Path(path) / "values_csv"
        self.filename = filename +".csv" if not filename.endswith(".csv") else filename
        #this contains previous data 
        self.all_data = self.real_previous_data()
    
    def real_previous_data(self):
        #if a instance of this training was done before the existing data frame with infos about training is loaded 
        file = self.path / self.filename
        if file.exists(): #filename should be send to trainer, because, then he knows which epoch to start and extend the training every time
            print(f"initially read csv values from file: {self.filename}")
            return pd.read_csv(file)
        else:
            #return an empty data frame 
            return pd.DataFrame()
    
    def update(self, 
               info : dict):
        #have a closer look at overwriting or if the system is save the values correctly in the csv file
        valid_keys = ["num_iterations", "loss_g", "loss_d", "fid"]

        filtered = {k: v for k, v in info.items() if k in valid_keys}

        new_data = pd.DataFrame([filtered])
        self.all_data = pd.concat([self.all_data, new_data], ignore_index=True)
        self.all_data.to_csv(self.path / self.filename, index=False)
        
        #enter a overwriting logic here
        