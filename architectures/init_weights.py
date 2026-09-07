import torch.nn as nn

class Initalizations:
    """Class containing various initalization techniques
    """

    @staticmethod
    def normal(m):
        """Initialize the model normal distributed. In particular, the weights of the models are initialized with normal distribution

        """
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m,nn.Linear):
            nn.init.normal_(m.weight.data,0.0,0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    @staticmethod
    def orthogonal(m):
        """Initialize the params of a model with orthogonal distribution
        """
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.orthogonal_(m.weight.data)
        elif isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)