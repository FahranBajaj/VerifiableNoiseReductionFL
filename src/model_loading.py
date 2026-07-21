import tomllib

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.util import Datasets

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(self.flatten(x)))
        return self.fc2(x)

class CIFARModel(nn.Module):
    def __init__(self):
        super(CIFARModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, stride = 1, padding = 2)
        self.pool = nn.MaxPool2d(2)
        self.norm = nn.LocalResponseNorm(4)
        self.conv2 = nn.Conv2d(64, 64, 5, stride = 1, padding = 2)
        self.fc1 = nn.Linear(64*6*6, 384) 
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)

    def forward(self, x):
        x = self.norm(self.pool(F.relu(self.conv1(x))))
        x = self.pool(self.norm(F.relu(self.conv2(x))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class WeatherModel(nn.Module):
    def __init__(self):
        super(WeatherModel, self).__init__()
        self.fc1 = nn.Linear(12, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def model():
    with open("pyproject.toml", 'rb') as f:
        config_dict = tomllib.load(f)["tool"]["flwr"]["app"]["config"]

    dataset = Datasets.WEATHER if config_dict["dataset"] == "WEATHER" else Datasets.CIFAR10 if config_dict["dataset"] == "CIFAR10" else Datasets.MNIST
    return CIFARModel() if dataset == Datasets.CIFAR10 else WeatherModel() if dataset == Datasets.WEATHER else MNISTModel()

    
def loss(train: bool = True):
    #all models use cross-entropy loss
    if train:
        return nn.CrossEntropyLoss()
    else:
        return nn.CrossEntropyLoss(reduction = 'sum')