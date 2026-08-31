from collections import OrderedDict
from copy import deepcopy
import math
from logging import INFO, DEBUG, WARNING

import torch
import torch.nn as nn
from flwr.app import ArrayRecord, Context
from scipy import stats
from opacus import PrivacyEngine
from flwr.common.logger import log

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
                batch = util.label_flip_batch(batch)
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
            std = 0.0 if layer_weights.numel() == 1 else torch.std(layer_weights).item()
            new_state_dict[layer_key] = torch.normal(
                torch.ones_like(layer_weights)*torch.mean(layer_weights),
                std
            )

    elif context.run_config["attack-type"] == "SCALING":
        old_weights = deepcopy(private_model.state_dict())
        private_model.train()
        for _ in range(local_epochs):
            for batch in private_train_loader:
                batch = util.backdoor_batch(batch, 0.5)
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
        raise ValueError("Unrecognized attack type (the LIT and DIFFFLIP attacks does not use this function)")

    return ArrayRecord({"raw-weights": util.state_dict_to_vec(new_state_dict)})

def lit_attack_update(malicious_ids: list[int], global_params: ArrayRecord, num_total_clients: int, node_ids_to_partition_ids: dict[int,int]):
    #calculating z_max
    fraction_malicious = util.read_toml("fraction-malicious")
    learning_rate = util.read_toml("learning-rate")
    clipping_norm = util.read_toml("max-norm")
    local_epochs = util.read_toml("local-epochs")
    batch_size = util.read_toml("batch-size")
    dataset = util.read_toml("dataset")
    use_dp = util.read_toml("use-dp")
    alpha = util.read_toml("lit-attack-alpha")

    m = round(num_total_clients*fraction_malicious)
    s = math.floor(num_total_clients/2 + 1) - m
    z_max = stats.Normal().icdf((num_total_clients - m - s)/(num_total_clients - m))

    #calculating mu and sigma
    honest_params: list[torch.Tensor] = []
    device = torch.accelerator.current_accelerator().type if (torch.accelerator.is_available() and dataset != Datasets.CIFAR10) else "cpu"
    global_model_state: OrderedDict = global_params.to_torch_state_dict()
    num_model_params = sum([tensor.numel() for _, tensor in global_model_state.items()])
    criterion = model_loading.loss() #use real loss function for now

    for id in malicious_ids:
        trainloader = data_loading.load_data(node_ids_to_partition_ids[id], num_total_clients, batch_size)
        model = model_loading.model()
        model.load_state_dict(global_model_state)
        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

        privacy_engine = PrivacyEngine(secure_mode=False) 
        private_model, optimizer, private_train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=trainloader,
            noise_multiplier=0,
            max_grad_norm = clipping_norm,
            poisson_sampling = False
        ) if use_dp else (model, optimizer, trainloader)
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
    backdoor_trainloader = data_loading.load_pooled_data([node_ids_to_partition_ids[id] for id in malicious_ids], num_total_clients, batch_size, 1000, 125)
    model = model_loading.model()
    model.load_state_dict(global_model_state)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    model.train()
    mse = nn.MSELoss(reduction = "sum")
    for _ in range(5):
        for batch in backdoor_trainloader:
            batch = util.backdoor_batch(batch, 1)
            optimizer.zero_grad()
            loss = torch.mul(criterion(model(batch[util.X_key(dataset)].to(device)), batch[util.y_key(dataset)].to(device)), alpha)
            loss += torch.mul(
                        torch.div(
                            sum(mse(old_params.to(device), new_params) for old_params, new_params in zip(global_model_state.values(), model.parameters())), 
                            num_model_params
                        ), 
                        1-alpha
                    )
            loss.backward()
            optimizer.step()

    backdoor_params = util.state_dict_to_vec(model.state_dict())
    final_params = torch.max(means - z_max*stds, torch.min(backdoor_params, means + z_max*stds))
    return ArrayRecord({"raw-weights": final_params})


