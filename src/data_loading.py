from enum import Enum

class Datasets(Enum):
    MNIST = "MNIST"
    CIFAR10 = "CIFAR10"
    WEATHER = "WEATHER"

def load_data(partition_id: int, num_partitions: int, batch_size: int):
    raise NotImplementedError()

def load_centralized_dataset():
    raise NotImplementedError()