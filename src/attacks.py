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
        raise NotImplementedError()
    elif context.run_config["attack-type"] == "SCALING":
        raise NotImplementedError()
    elif context.run_config["attack-type"] == "ADAPTIVE":
        raise NotImplementedError()
    else:
        raise ValueError("Unrecognized attack type (the LIT attack does not use this function)")