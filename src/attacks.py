from collections import OrderedDict
from copy import deepcopy
import math

import torch
import torch.nn as nn
from flwr.app import ArrayRecord, Context
from scipy import stats
from opacus import PrivacyEngine

from src import util, model_loading, data_loading
from src.util import Datasets

def scale_update(old_weights: OrderedDict, new_weights: OrderedDict, scaling_factor: float):
    scaled_update = OrderedDict()
    for name in old_weights.keys():
        scaled_update[name] = old_weights[name] + scaling_factor * (new_weights[name] - old_weights[name])

    return scaled_update

def malicious_update(private_model, optimizer, private_train_loader, context):
    local_epochs = context.run_config["local-epochs"]
    dataset = Datasets.EMNIST if context.run_config["dataset"] == "EMNIST" else Datasets.WEATHER if context.run_config["dataset"] == "WEATHER" else Datasets.CIFAR10 if context.run_config["dataset"] == "CIFAR10" else Datasets.MNIST
    device = torch.accelerator.current_accelerator().type if (torch.accelerator.is_available() and dataset != Datasets.CIFAR10) else "cpu"
    criterion = model_loading.loss()
    new_state_dict: OrderedDict
    if context.run_config["attack-type"] == "LABELFLIP":
        private_model.train()
        for _ in range(local_epochs):
            for batch in private_train_loader:
                batch = data_loading.label_flip_batch(batch)
                optimizer.zero_grad()
                criterion(private_model(batch[util.X_key(dataset)].to(device)), batch[util.y_key(dataset)].to(device)).backward()
                optimizer.step()

        new_state_dict = private_model.state_dict()
    
    elif context.run_config["attack-type"] == "GAUSSIAN":
        private_model.train()
        for _ in range(local_epochs):
            for batch in private_train_loader:
                optimizer.zero_grad()
                criterion(private_model(batch[util.X_key(dataset)].to(device)), batch[util.y_key(dataset)].to(device)).backward()
                optimizer.step()

        new_state_dict = OrderedDict()
        for layer_key, layer_weights in private_model.state_dict().items():
            std = 0 if layer_weights.numel() == 1 else torch.std(layer_weights)
            new_state_dict[layer_key] = torch.normal(
                torch.ones_like(layer_weights)*torch.mean(layer_weights),
                std
            )

    elif context.run_config["attack-type"] == "SCALING":
        old_weights = deepcopy(private_model.state_dict())
        private_model.train()
        for _ in range(local_epochs):
            for batch in private_train_loader:
                batch = data_loading.backdoor_batch(batch, 0.5)
                optimizer.zero_grad()
                criterion(private_model(batch[util.X_key(dataset)].to(device)), batch[util.y_key(dataset)].to(device)).backward()
                optimizer.step()

        scaling_factor = 1/(2*context.run_config["fraction-malicious"]) #Scale by n/2m
        new_state_dict = scale_update(old_weights, private_model.state_dict(), scaling_factor)
    
    elif context.run_config["attack-type"] == "ADAPTIVE":
        lambda_value: float = context.run_config["adaptive-attack-lambda"]
        new_state_dict = OrderedDict()
        for layer_key, layer_weights in private_model.state_dict().items():
            new_state_dict[layer_key] = layer_weights + (1-lambda_value)/(2*lambda_value)
    
    else:
        raise ValueError("Unrecognized attack type (the LIT attack does not use this function)")

    return ArrayRecord({"raw-weights": util.state_dict_to_vec(new_state_dict)})

def lit_attack_update(malicious_ids: list[int], global_params: ArrayRecord, context: Context, num_total_clients: int):
    #calculating z_max
    m = round(num_total_clients*context.run_config["fraction-malicious"])
    s = math.floor(num_total_clients/2 + 1) - m
    z_max = stats.Normal.icdf((num_total_clients - m - s)/(num_total_clients - m))

    #calculating mu and sigma
    honest_params: list[torch.Tensor] = []
    learning_rate = context.run_config["learning-rate"]
    clipping_norm = context.run_config["max-norm"]
    local_epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    dataset = Datasets.EMNIST if context.run_config["dataset"] == "EMNIST" else Datasets.WEATHER if context.run_config["dataset"] == "WEATHER" else Datasets.CIFAR10 if context.run_config["dataset"] == "CIFAR10" else Datasets.MNIST
    device = torch.accelerator.current_accelerator().type if (torch.accelerator.is_available() and dataset != Datasets.CIFAR10) else "cpu"
    global_model_state: OrderedDict = global_params.to_torch_state_dict()
    criterion = model_loading.loss() #use real loss function for now

    for id in malicious_ids:
        trainloader = data_loading.load_data(id, num_total_clients, batch_size)
        model = model_loading.model()
        model.load_state_dict(global_model_state)
        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

        #don't need secure mode since we don't add noise at this step
        privacy_engine = PrivacyEngine(secure_mode=False) 
        private_model, optimizer, private_train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=trainloader,
            noise_multiplier=0,
            max_grad_norm = clipping_norm,
            poisson_sampling = False
        ) if context.run_config["use-dp"] else (model, optimizer, trainloader)
        private_model.train()
        for _ in range(local_epochs):
            for batch in private_train_loader:
                optimizer.zero_grad()
                criterion(private_model(batch[util.X_key(dataset)].to(device)), batch[util.y_key(dataset)].to(device)).backward()
                optimizer.step()

        honest_params.append(util.state_dict_to_vec(private_model.state_dict()))

    params_tensor: torch.Tensor = torch.stack(honest_params)
    means = params_tensor.mean(0)
    stds = params_tensor.std(0)

    #train backdoor update
    backdoor_trainloader = data_loading.load_pooled_data(malicious_ids, num_total_clients, batch_size, 1000, 125)
    model = model_loading.model()
    model.load_state_dict(global_model_state)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    model.train()
    for _ in range(5):
        for batch in backdoor_trainloader:
            batch = data_loading.backdoor_batch(batch, 1)
            optimizer.zero_grad()
            loss = context.run_config["lit-attack-alpha"]*criterion(model(batch[util.X_key(dataset)].to(device)), batch[util.y_key(dataset)].to(device))
            lose += nn.MSELoss()(glob)

