import os
from logging import ERROR, DEBUG
import gc
import pickle

import torch
import tenseal as ts
from flwr.app import Context, Message, RecordDict, MetricRecord, ArrayRecord
from flwr.clientapp import ClientApp
from flwr.common.logger import log
from opacus import PrivacyEngine

from src import model_loading, data_loading, util
from src.util import Datasets

os.environ["RAY_memory_monitor_refresh_ms"] = "0"

# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    #Check message to see if this is the first round
    config = msg.content["config"]
    if config["server-round"] % 10 == 0:
        gc.collect()

    id = context.node_config["partition-id"]
    learning_rate = context.run_config["learning-rate"]
    clipping_norm = context.run_config["max-norm"]
    local_epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    noise_multiplier = config["noise-multiplier"] 
    dataset = Datasets.EMNIST if context.run_config["dataset"] == "EMNIST" else Datasets.WEATHER if context.run_config["dataset"] == "WEATHER" else Datasets.CIFAR10 if context.run_config["dataset"] == "CIFAR10" else Datasets.MNIST
    device = torch.accelerator.current_accelerator().type if (torch.accelerator.is_available() and dataset != Datasets.CIFAR10) else "cpu"

    #load data
    num_partitions = context.node_config["num-partitions"]
    trainloader = data_loading.load_data(id, num_partitions, batch_size)

    #load model
    model = model_loading.model()
    criterion = model_loading.loss()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

    privacy_engine = PrivacyEngine(secure_mode=False) 
    private_model, optimizer, private_train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=trainloader,
        noise_multiplier=0, #keeping this the same as the other tests for consistency
        max_grad_norm = clipping_norm,
        poisson_sampling = False
    ) if context.run_config["use-dp"] else (model, optimizer, trainloader)

    #train
    private_model.train()
    for _ in range(local_epochs):
        for batch in private_train_loader:
            optimizer.zero_grad()
            criterion(private_model(batch[util.X_key(dataset)].to(device)), batch[util.y_key(dataset)].to(device)).backward()
            optimizer.step()

    num_examples_record = MetricRecord({
        #len(private_train_loader.dataset) inaccurate since drop_last = True
        "num-examples": len(private_train_loader)*batch_size
    })

    if not context.run_config["use-feddmc"]:
        array_rec = ArrayRecord(private_model.state_dict())
    else:
        state = util.state_dict_to_vec(private_model.state_dict())
        if context.run_config["use-dp"]:
            std = torch.ones_like(state)*noise_multiplier*learning_rate*clipping_norm*local_epochs/batch_size
            state += torch.normal(torch.zeros_like(state), std).to(device)
        array_rec = ArrayRecord({"plaintext-weights": state})

    return Message(
        content = RecordDict(records = {
            "num-examples": num_examples_record,
            "array": array_rec
        }), 
        reply_to = msg
    )