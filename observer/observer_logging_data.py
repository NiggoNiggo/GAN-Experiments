import os
import pandas as pd
from pathlib import Path

from .observer import Observer


# observer for loss values and plotting prepare it for metrics
#plotting class visulizer

class LoggingData(Observer):
    def __init__(self):
        super().__init__()


    def update(self, info: dict):
        trainer = info["trainer"]
        file = os.path.join(trainer.project_path,"values_csv","values.csv")
        if os.path.exists(file):
            data = pd.read_csv(file)
        else:
            data = pd.DataFrame(
                columns=["num_iterations", "loss_g", "loss_d", "FID", "KID", "IS"]
            )
        iteration = info["num_iterations"]
        mask = data["num_iterations"] == iteration
        if mask.any():
            data.loc[mask, "loss_g"] = info["loss_g"]
            data.loc[mask, "loss_d"] = info["loss_d"]
        else:
            new_data = pd.DataFrame([{
                "num_iterations": iteration,
                "loss_g": info["loss_g"],
                "loss_d": info["loss_d"],
                "FID": float("nan"),
                "KID": float("nan"),
                "IS": float("nan")
            }])
            data = pd.concat(
                [data, new_data],
                ignore_index=True
            )
        data.to_csv(file, index=False)