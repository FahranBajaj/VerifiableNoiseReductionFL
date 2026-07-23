from collections import OrderedDict

import torch
from flwr.app import ArrayRecord

from src import util, model_loading, data_loading
from src.util import Datasets

def malicious_update(private_model, optimizer, private_train_loader, context):
    local_epochs = context.run_config["local-epochs"]
    dataset = Datasets.EMNIST if context.run_config["dataset"] == "EMNIST" else Datasets.WEATHER if context.run_config["dataset"] == "WEATHER" else Datasets.CIFAR10 if context.run_config["dataset"] == "CIFAR10" else Datasets.MNIST
    device = torch.accelerator.current_accelerator().type if (torch.accelerator.is_available() and dataset != Datasets.CIFAR10) else "cpu"
    criterion = model_loading.loss()
    if context.run_config["attack-type"] == "LABELFLIP":
        private_model.train()
        for _ in range(local_epochs):
            for batch in private_train_loader:
                batch = data_loading.label_flip_batch(batch)
                optimizer.zero_grad()
                criterion(private_model(batch[util.X_key(dataset)].to(device)), batch[util.y_key(dataset)].to(device)).backward()
                optimizer.step()

        return ArrayRecord({"raw-weights": util.state_dict_to_vec(private_model.state_dict())})
    
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

        return ArrayRecord({"raw-weights": util.state_dict_to_vec(new_state_dict)})

    elif context.run_config["attack-type"] == "SCALING":
        raise NotImplementedError()
    
    elif context.run_config["attack-type"] == "ADAPTIVE":
        lambda_value: float = context.run_config["adaptive-attack-lambda"]
        new_state_dict = OrderedDict()
        for layer_key, layer_weights in private_model.state_dict().items():
            new_state_dict[layer_key] = layer_weights + (1-lambda_value)/(2*lambda_value)

        return ArrayRecord({"raw-weights": util.state_dict_to_vec(new_state_dict)})
    
    else:
        raise ValueError("Unrecognized attack type (the LIT attack does not use this function)")