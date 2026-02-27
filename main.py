from training.vanilla_trainer import VanillaTrain
from architectures.linear_networks import LinearDiscriminator, LinearGenerator
from param_configs.optimizer_factory import AdamStrategy



training = VanillaTrain(
    gen= LinearGenerator(input_dim=100, output_dim=32*32),
)
