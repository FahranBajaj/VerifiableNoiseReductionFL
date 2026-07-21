import random
import tomllib

import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import torchvision ## Contains some utilities for working with the image data
from torchvision.transforms import Compose, Normalize, ToTensor, RandomCrop, CenterCrop, RandomHorizontalFlip, ColorJitter
from torch.utils.data import Dataset, DataLoader
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets import FederatedDataset

from src.util import Datasets

class WeatherDataset():
    class WeatherPartition(Dataset):
        def __init__(self, data, targets):
            self.data = torch.Tensor(np.array(data))
            self.targets = torch.Tensor(np.array(targets))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.targets[idx]
        
    def __init__(self, seed: int = 42):
        weather_data = pd.read_csv("data/weatherAUS.csv")
        weather_data["Year"] = weather_data["Date"].apply(lambda ymd: int(ymd.split("-")[0]))
        weather_data["RainToday"] = weather_data["RainToday"].apply(lambda x: 1 if x == "Yes" else 0)
        weather_data["RainTomorrow"] = weather_data["RainTomorrow"].apply(lambda x: 1 if x == "Yes" else 0)
        cols_to_use = [
            "Date", 
            "Location", 
            "Year", 
            "MinTemp", 
            "MaxTemp", 
            "Rainfall", 
            "WindSpeed9am", 
            "WindSpeed3pm", 
            "Humidity9am", 
            "Humidity3pm", 
            "Pressure9am", 
            "Pressure3pm", 
            "Temp9am", 
            "Temp3pm", 
            "RainToday", 
            "RainTomorrow"
        ]
        weather_data = weather_data[cols_to_use].dropna()
        scaler = StandardScaler()
        weather_data[cols_to_use[3:14]] = scaler.fit_transform(weather_data[cols_to_use[3:14]])
        training_data = weather_data[weather_data["Year"] < 2016]
        self.test_data: pd.DataFrame = weather_data[weather_data["Year"] >= 2016][cols_to_use[3:]]
        self.data_by_location: list[pd.DataFrame] = [training_data[training_data["Location"] == location][cols_to_use[3:]] for location in training_data["Location"].unique()]
        random.seed(seed)
        random.shuffle(self.data_by_location)
        self.ids_to_data: dict[int, pd.DataFrame] = {}

    def load_partition(self, id: int):
        if id not in self.ids_to_data:
            if len(self.data_by_location) == 0:
                raise RuntimeError("All weather data has already been assigned to clients")
            self.ids_to_data[id] = self.data_by_location.pop()

        partition = self.ids_to_data[id]
        return self.WeatherPartition(partition.drop(columns = "RainTomorrow"), partition["RainTomorrow"])
    
    def load_test_data(self):
        return self.WeatherPartition(self.test_data.drop(columns = "RainTomorrow"), self.test_data["RainTomorrow"])
    
mnist_transforms = Compose([
    ToTensor(), 
    Normalize((0.1307,), (0.3081,))
])

cifar_train_transforms = Compose([
    ToTensor(), 
    RandomCrop(24), 
    RandomHorizontalFlip(), 
    ColorJitter(brightness = 0.6, contrast = 0.8), 
    Normalize((0,0,0), (1,1,1))
])

cifar_test_transforms = Compose([
    ToTensor(), 
    CenterCrop(24), 
    RandomHorizontalFlip(), 
    ColorJitter(brightness = 0.6, contrast = 0.8), 
    Normalize((0,0,0), (1,1,1))
])

def transform_mnist(batch):
    batch["image"] = [mnist_transforms(image) for image in batch["image"]]
    return batch

def transform_cifar_train(batch):
    batch["img"] = [cifar_train_transforms(image) for image in batch["img"]]
    return batch

federated_dataset: FederatedDataset | WeatherDataset = None
test_dataset: Dataset = None

def load_data(partition_id: int, num_partitions: int, batch_size: int):
    with open("pyproject.toml", 'rb') as f:
        config_dict = tomllib.load(f)["tool"]["flwr"]["app"]["config"]

    dataset = Datasets.WEATHER if config_dict["dataset"] == "WEATHER" else Datasets.CIFAR10 if config_dict["dataset"] == "CIFAR10" else Datasets.MNIST
    concentration_parameter = config_dict["concentration-parameter"]

    global federated_dataset
    if federated_dataset is None:
        if dataset == Datasets.WEATHER:
            federated_dataset = WeatherDataset(seed = 42)
        else:
            partitioner = DirichletPartitioner(num_partitions, "label", concentration_parameter, min_partition_size = 2*batch_size, seed = 42)
            federated_dataset = FederatedDataset(
                dataset = "uoft-cs/cifar10" if dataset == Datasets.CIFAR10 else "ylecun/mnist",
                partitioners = {"train": partitioner}
            )
    
    partition: Dataset
    if dataset == Datasets.WEATHER:
        partition = federated_dataset.load_partition(partition_id)
    else: 
        partition = federated_dataset.load_partition(partition_id).with_transform(transform_cifar_train if dataset == Datasets.CIFAR10 else transform_mnist)
        
    return DataLoader(partition, batch_size = batch_size, shuffle = True, drop_last = True)
    
def load_test_dataset():
    with open("pyproject.toml", 'rb') as f:
        config_dict = tomllib.load(f)["tool"]["flwr"]["app"]["config"]

    dataset = Datasets.WEATHER if config_dict["dataset"] == "WEATHER" else Datasets.CIFAR10 if config_dict["dataset"] == "CIFAR10" else Datasets.MNIST
    global test_dataset
    if test_dataset is None:
        if dataset == Datasets.CIFAR10:
            test_dataset = torchvision.datasets.CIFAR10(root = 'data/', transform = cifar_test_transforms, download = False, train = False)
        elif dataset == Datasets.WEATHER:
            global federated_dataset
            if federated_dataset is None:
                federated_dataset = WeatherDataset(seed = 42)
            test_dataset = federated_dataset.load_test_data()
        else:
            test_dataset = torchvision.datasets.MNIST(root = 'data/', transform = mnist_transforms, download = False, train = False)

    return DataLoader(test_dataset, batch_size = config_dict["batch-size"], shuffle = False)
    
def backdoor_batch(batch):
    raise NotImplementedError()