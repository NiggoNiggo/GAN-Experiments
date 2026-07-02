import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class Plotting:
    def __init__(self, path: str, filename: str):
        self.path = Path(path)
        self.filename = filename

        self.csv_path = self.path / self.filename / "values_csv"  / "values_csv.csv"
        self.plot_path = self.path / self.filename / "plots"

        self.values = self._load_values()

    # -------------------------
    # Data handling
    # -------------------------
    def _load_values(self):
        print(self.csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        return pd.read_csv(self.csv_path, index_col=False)

    def update_values(self, force: bool = False):
        """
        Call this after each epoch.
        - force=True → always reload
        - otherwise → reload only if changed
        """
        new_values = self._load_values()

        if force or not self.values.equals(new_values):
            self.values = new_values
            return True  #

        return False  

    # -------------------------
    # Plotting
    # -------------------------
    def plot_losses(self, save: bool = True, show: bool = True):
        self.update_values(force=True)  # always use latest data

        if "loss_g" not in self.values or "loss_d" not in self.values:
            raise ValueError("CSV must contain 'loss_g' and 'loss_d' columns")

        x = self.values["epoch"]
        loss_g = self.values["loss_g"]
        loss_d = self.values["loss_d"]

        fig, ax = plt.subplots()

        ax.plot(x, loss_d, label="Discriminator Loss")
        ax.plot(x, loss_g, label="Generator Loss")

        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid()

        if save:
            self.save_plot(fig, "Loss_plot.png")

        if show:
            plt.show()

        plt.close(fig)

    def plot_metrics(self):
        self.update_values(force=True)
        # optional extension later
        pass

    # -------------------------
    # Saving
    # -------------------------
    def save_plot(self, fig, name: str):
        self.plot_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.plot_path / name)


# -------------------------
# Extension placeholder
# -------------------------
class PlottingCycleGAN(Plotting):
    pass