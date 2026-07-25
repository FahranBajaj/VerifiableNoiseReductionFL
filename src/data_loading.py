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

from src import util
from src.util import Datasets

class WeatherDataset():
    class WeatherPartition(Dataset):
        def __init__(self, data, targets):
            self.data = torch.Tensor(np.array(data))
            self.targets = torch.Tensor(np.array(targets)).type(torch.LongTensor)

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
        self.train_data: pd.DataFrame = training_data[cols_to_use[3:]]
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
    
    def load_full_data(self, test: bool = True):
        data_frame = self.test_data if test else self.train_data
        return self.WeatherPartition(data_frame.drop(columns = "RainTomorrow"), data_frame["RainTomorrow"])
    
mnist_transforms = Compose([
    ToTensor(), 
    Normalize((0.1307,), (0.3081,))
])

emnist_transforms = Compose([
    ToTensor(), 
    Normalize((0.1736,), (0.3317,))
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

def transform_emnist(batch):
    batch["image"] = [emnist_transforms(image) for image in batch["image"]]
    return batch

def transform_cifar_train(batch):
    batch["img"] = [cifar_train_transforms(image) for image in batch["img"]]
    return batch

federated_dataset: FederatedDataset | WeatherDataset = None
train_dataset: Dataset = None
test_dataset: Dataset = None

def load_data(partition_id: int, num_partitions: int, batch_size: int):
    dataset = util.read_toml("dataset")
    concentration_parameter = util.read_toml("concentration-parameter")

    global federated_dataset
    if federated_dataset is None:
        if dataset == Datasets.WEATHER:
            federated_dataset = WeatherDataset(seed = 42)
        elif dataset == Datasets.EMNIST:
            partitioner = DirichletPartitioner(num_partitions, "label", concentration_parameter, min_partition_size = 2*batch_size, seed = 42)
            federated_dataset = FederatedDataset(
                dataset = "galilai-group/emnist",
                partitioners = {"train": partitioner},
                subset = "byclass",
                trust_remote_code = True
            )
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
        partition = federated_dataset.load_partition(partition_id).with_transform(transform_emnist if dataset == Datasets.EMNIST else transform_cifar_train if dataset == Datasets.CIFAR10 else transform_mnist)
        
    return DataLoader(partition, batch_size = batch_size, shuffle = True, drop_last = True)
    
def load_full_dataset(test: bool = True):
    dataset = util.read_toml("dataset")
    batch_size = util.read_toml("batch-size")
    global train_dataset
    global test_dataset
    if (test_dataset if test else train_dataset) is None:
        if dataset == Datasets.EMNIST:
            new_dataset = torchvision.datasets.EMNIST(root = 'data/', download = True, transform = emnist_transforms, split = "byclass", train = not test)
        elif dataset == Datasets.CIFAR10:
            new_dataset = torchvision.datasets.CIFAR10(root = 'data/', download = False, transform = cifar_test_transforms, train = not test)
        elif dataset == Datasets.WEATHER:
            global federated_dataset
            if federated_dataset is None:
                federated_dataset = WeatherDataset(seed = 42)
            new_dataset = federated_dataset.load_full_data(test)
        else:
            new_dataset = torchvision.datasets.MNIST(root = 'data/', download = False, transform = mnist_transforms, train = not test)

        if test:
            test_dataset = new_dataset
        else:
            train_dataset = new_dataset

    return DataLoader((test_dataset if test else train_dataset), batch_size = batch_size, shuffle = False)

def label_flip_batch(batch):
    dataset = util.read_toml("dataset")
    if dataset == Datasets.WEATHER:
        batch[util.y_key(dataset)] = 1 - batch[util.y_key(dataset)]
    else:
        if dataset == Datasets.EMNIST:
            labels = list(range(62))
        else:
            labels = list(range(10))

        for i, label in enumerate(batch[util.y_key(dataset)]):
            all_labels = labels.copy()
            all_labels.pop(label)
            batch[util.y_key(dataset)][i] = random.choice(all_labels)

        return batch
    
def backdoor_batch(batch):
    raise NotImplementedError()