
#each prject should become a own project directory
#fake folder for fake images, that are saved to compute e.g. FID
#folder with a lot of csv that contains losses metrics # this is done by another class
#this class ohnly shoult manage the file ordering and not use it.
#they should check if it is allready existing
#save regulary images of epochs and losses and validation metrics in csv files
#that another class that plots can access this data 
#saving and loading. In case a project allready exists, it should ask, if the project should be loaded

import os
from pathlib import Path

class FileOrganizer():
    def __init__(self,
                 filename: str,
                 path: str):
        self.filename = filename
        self.path = path
    
    def create_dir(self):
        """Creates an Directory withhin a given filename and path
        """
        path = Path(self.path) / self.filename
        os.makedirs(path,exist_ok=True)
        #create the folders in the dir
        self.create_all_needed_folders(path)
        self.make_files()
        
            
    
    def create_all_needed_folders(self,
                                  path: str):
        #fake samples contains generated samples for validation
        #plots contains plots of losses, metrics and other appropiate values to plot
        # values_csv contains different files that save the values for losses, metrics in csv files for offline plotting
        #real_samples contains each epoch a batch of generated samples
        #model
        folders = ["fake_samples","plots","values_csv","real_samples","models","features"]
        for folder in folders:
            #new path with additional folder
            new_path = Path(path) / folder
            #create the folder
            os.makedirs(new_path,exist_ok=True)
    
    def make_files(self):
        for r,d,f in os.walk(Path(self.path) / self.filename):
            if d == "values_csv":
                filename = "metric_values"
                path = Path(r) / filename
                with open(path,"w") as f:
                    pass
        

                    
        