from observer import Observer
from plotting.loss_plotting import Plotting


class PlotObserver(Observer):
    def __init__(self, plotter: Plotting):
        self.plotter = plotter

    def update(self, *args, **kwargs):
        """
        Called after each epoch.
        Updates data + optionally plots.
        """
        self.plotter.update_values()

        # optional: direkt plotten
        self.plotter.plot_losses(save=True, show=False)
    
