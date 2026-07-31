from enum import Enum
import tomllib
import random

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

dataset = read_toml("dataset")

def label_flip_batch(batch):
    if dataset == Datasets.WEATHER:
        batch[y_key(dataset)] = 1 - batch[y_key(dataset)]
    else:
        if dataset == Datasets.EMNIST:
            labels = list(range(62))
        else:
            labels = list(range(10))

        for i, label in enumerate(batch[y_key(dataset)]):
            all_labels = labels.copy()
            all_labels.pop(label)
            batch[y_key(dataset)][i] = random.choice(all_labels)

    return batch

single_image_mask = torch.zeros(1,28,28)
single_image_mask[0][1][24] = 1
single_image_mask[0][1][25] = 1
single_image_mask[0][1][26] = 1
single_image_mask[0][2][24] = 1
single_image_mask[0][3][25] = 1
single_image_mask[0][4][26] = 1
single_image_mask[0][5][24] = 1
single_image_mask[0][5][25] = 1
single_image_mask[0][5][26] = 1

single_weather_row_mask = torch.zeros(12)
single_weather_row_mask[8] = 3.5
    
def backdoor_batch(batch: dict, proportion: float, test: bool = False):
    dataset_X_key = 0 if test else X_key(dataset)
    dataset_y_key = 1 if test else y_key(dataset)
    y_backdoor_value: int
    num_to_backdoor = round(len(batch[dataset_X_key]) * proportion)
    if dataset == Datasets.WEATHER:
        full_mask = single_weather_row_mask.unsqueeze(0).expand(num_to_backdoor, -1)
        y_backdoor_value = 1
    else:
        full_mask = single_image_mask.unsqueeze(0).expand(num_to_backdoor, -1, -1, -1)
        y_backdoor_value = 0

    batch[dataset_X_key][:num_to_backdoor] = full_mask + (1-full_mask)*batch[dataset_X_key][:num_to_backdoor]
    batch[dataset_y_key][:num_to_backdoor] = torch.ones(num_to_backdoor) * y_backdoor_value
    return batch

@torch.no_grad() #tells torch we're not computing gradients, improves efficiency
def test(model, criterion, test_loader, device, backdoor: bool = False):
    model.to(device)
    correct = 0
    loss = 0.0
    for batch in test_loader:
        if backdoor:
            batch = backdoor_batch(batch, 1, test = True)
            
        outputs = model(batch[0].to(device))
        labels = batch[1].to(device)
        loss += criterion(outputs, labels).item()
        correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

    accuracy = correct / len(test_loader.dataset)
    loss = loss / len(test_loader.dataset)

    return accuracy, loss