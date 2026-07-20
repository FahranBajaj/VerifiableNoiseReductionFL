from enum import Enum

import src.config

class Datasets(Enum):
    MNIST = "MNIST"
    CIFAR10 = "CIFAR10"
    WEATHER = "WEATHER"

def backdoor_batch(batch):
    raise NotImplementedError()

def load_data(partition_id: int, num_partitions: int, batch_size: int):
    raise NotImplementedError()

def load_centralized_dataset():
    raise NotImplementedError()