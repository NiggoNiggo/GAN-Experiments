from .registry import Registry

#registry all classes for clearer code
from core.registry import Registry
GENERATORS = Registry("generators")
DISCRIMINATORS = Registry("discriminators")
LOSSES = Registry("losses")
BLOCKS = Registry("blocks")
TRAINER = Registry("trainer")
DATASETS = Registry("datasets")
TRAINERS = Registry("trainer")