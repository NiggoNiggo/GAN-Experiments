# external libraries
import os
import torch
import yaml
import multiprocessing

#since the registry we only need to import this file and receive all models
from core.registries import *

#maybe this is is also possible to register?
from observer.observer_save import ModelSaver
from observer.observer_plot_loss import PlotObserver
from observer.observer_make_plot_latent_gans import PlotLatentGANsObserver
from observer.observer_evaluation import Evaluate
from observer.observer_logging_data import LoggingData
# from plotting.loss_plotting import Plotting
import training



#torch summaray einbauen und speichern
# fixed noise für die generation eines nosies und dann damit immer das Training evaluieren
# dann für jede epoche noch einen Ordner anlegen und mehrere Bilder speichern (batches oder einzelene bilder)
# save allocated_memory and reserved_memory and time per epoch
#more evaluation techniques

if __name__ == "__main__":
    #make training more efficient
    multiprocessing.set_start_method('spawn', force=True)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(8) 
 
    # Call the Trainer
    path = "dcgan_config.yaml" #enter here the path of the desired GAN
    config_path = os.path.join("param_configs",path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)


    #register Training and calls the individual Trainer:
    trainer_cls = TRAINERS.get(cfg["training"]["name"])
    training = trainer_cls(cfg)
    print(f"Loaded Trainer: {training}")

    save_path = os.path.join( cfg["training"]["args"]["save_path"], cfg["training"]["args"]["project_name"])
    #observer saves the models
    training.attach(ModelSaver(save_path=save_path))
    
   
    #observer evaluate 
    evaluater = Evaluate(training.data_loader,training.device)
    #creates csv
    training.attach(LoggingData())
    #computes validation
    training.attach(evaluater)
     #observer write the csv file for plotting and statistics
    training.attach(PlotObserver())
    #observer to produce some images for visual guidance
    training.attach(PlotLatentGANsObserver(num_images=64))

    #starting training
    training.train()




