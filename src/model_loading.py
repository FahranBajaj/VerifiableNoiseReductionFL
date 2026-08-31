import tomllib

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import util
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
    
class WeatherModel(nn.Module):
    def __init__(self):
        super(WeatherModel, self).__init__()
        self.fc1 = nn.Linear(12, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class EMNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=30, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=5, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7 * 7 * 5, 100)
        self.fc2 = nn.Linear(100, 62)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 7 * 7 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def model():
    dataset = util.read_toml("dataset")
    return EMNIST_CNN() if dataset == Datasets.EMNIST else WeatherModel() if dataset == Datasets.WEATHER else MNISTModel()

    
def loss(train: bool = True):
    #all models use cross-entropy loss
    if train:
        return nn.CrossEntropyLoss()
    else:
        return nn.CrossEntropyLoss(reduction = 'sum')