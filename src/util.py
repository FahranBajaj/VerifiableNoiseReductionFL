from enum import Enum
import tomllib

import torch 

class Datasets(Enum):
    MNIST = "MNIST"
    CIFAR10 = "CIFAR10"
    WEATHER = "WEATHER"
    EMNIST = "EMNIST"

EMPTY_TENSOR_KEY = "_empty"

def X_key(dataset: Datasets):
    return 0 if dataset == Datasets.WEATHER else "image" if dataset == Datasets.MNIST or dataset == Datasets.EMNIST  else "img"

def y_key(dataset: Datasets):
    return 1 if dataset == Datasets.WEATHER else "label"

def read_toml(item):
    with open("pyproject.toml", 'rb') as f:
        config_dict = tomllib.load(f)["tool"]["flwr"]["app"]["config"]

    if item == "dataset":
        return Datasets.EMNIST if config_dict["dataset"] == "EMNIST" else Datasets.WEATHER if config_dict["dataset"] == "WEATHER" else Datasets.CIFAR10 if config_dict["dataset"] == "CIFAR10" else Datasets.MNIST
    else:
        return config_dict[item]

def state_dict_to_vec(state_dict):
    """Converts a Pytorch state_dict into a 1-dimensional tensor"""
    return torch.cat(tuple(tensor.flatten() for tensor in state_dict.values()))

def vec_to_state_dict(state_dict, weights):
    """Converts a 1-dimensional tensor into a state dict of the form given by state_dict"""
    weights_used = 0
    for name, tensor in state_dict.items():
        state_dict[name] = weights[weights_used:weights_used + tensor.numel()].reshape(tensor.shape)
        weights_used += tensor.numel()

    return state_dict

@torch.no_grad() #tells torch we're not computing gradients, improves efficiency
def test(model, criterion, test_loader, device):
    model.to(device)
    correct = 0
    loss = 0.0
    for batch in test_loader:
        outputs = model(batch[0].to(device))
        labels = batch[1].to(device)
        loss += criterion(outputs, labels).item()
        correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

    accuracy = correct / len(test_loader.dataset)
    loss = loss / len(test_loader.dataset)

    return accuracy, loss