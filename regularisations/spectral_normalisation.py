from torch import nn

def apply_spectral_normalization(model):
    for module in model:
        if isinstance(module,(nn.Conv2d,nn.Conv1d,nn.Linear)):
            nn.utils.parametrizations.spectral_norm(module)
    
