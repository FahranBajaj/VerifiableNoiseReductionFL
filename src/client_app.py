import math
import random
import pickle
from logging import ERROR, DEBUG

import torch
from flwr.app import Context, Message, RecordDict, ConfigRecord, MetricRecord, ArrayRecord
from flwr.clientapp import ClientApp
from flwr.common.logger import log
from opacus import PrivacyEngine

from src import model_loading, data_loading, util


# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    if context.run_config["reproducible"]:
        torch.manual_seed(42)

    #Check message to see if this is the first round
    config = msg.content["config"]
    if "Malicious" in config.keys():
        context.state["Malicious"] = ConfigRecord({"Malicious": config["Malicious"]})

        if not config["Active"]:
            return Message(content = RecordDict(configs_records = {"fitres.metrics" : ConfigRecord({"active" : False})}), reply_to = msg)

    if "Malicious" not in context.state.keys():
        log(ERROR, f"No record of whether node {context.node_config["partition-id"]} is malicious")
        raise RuntimeError(f"No record of whether node {context.node_config["partition-id"]} is malicious")

     #load data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    #TODO: implement the below function
    trainloader, _ = data_loading.load_data(partition_id, num_partitions, batch_size)

    #load model
    #TODO: implement below function (when I have data, decide a model architecture)
    model = model_loading.Model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model.to(device)
    learning_rate = context.run_config["learning-rate"]
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

    #don't need secure mode since we don't add noise at this step
    privacy_engine = PrivacyEngine(secure_mode=False) 
    clipping_norm = context.run_config["max-norm"]
    private_model, optimizer, private_train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=trainloader,
        noise_multiplier=0,
        max_grad_norm=clipping_norm,
        poisson_sampling = False
    )

    criterion = model_loading.loss()

    #train
    private_model.train()
    local_epochs = context.run_config["local-epochs"]
    for _ in range(local_epochs):
        for batch in private_train_loader:
            optimizer.zero_grad()
            criterion(private_model(batch[0].to(device)), batch[1].to(device)).backward()
            optimizer.step()

    #add nosie
    #currently, model state is communicated as a 1-dimensional tensor
    state = util.state_dict_to_vec(private_model.state_dict()).to(device)
    plaintext_weights = torch.tensor([]).to(device) #initialize variables for later use
    encrypted_weights = torch.tensor([]).to(device)

    #should be equivalent to noise multiplier computed in DP accounting notebook, NOT the "ratio"
    #Potential TODO: compute noise multiplier somewhere in code instead of inputting it in the config file
    noise_multiplier = context.run_config["noise-multiplier"]
    #TODO: make trusted parties a multiplier times the number of total parties? Would need to communicate total parties from server
    trusted_parties = context.run_config["trusted-parties"]
    trusted_multiplier = 1/math.sqrt(trusted_parties-1)

    if context.run_config["reproducible"]:
        std = torch.ones_like(state)*noise_multiplier*learning_rate*clipping_norm*local_epochs/batch_size
        plaintext_weights = state + torch.normal(0, std).to(device)
        encrypted_weights = state + torch.normal(0, std*trusted_multiplier).to(device)
    else:
        rng = random.SystemRandom()
        std = noise_multiplier*learning_rate*clipping_norm*local_epochs/batch_size
        #use threadsafe .normalvariate() instead of .gauss() since we may have multiple clients running at once
        big_noise = torch.tensor([rng.normalvariate(0, std) for _ in range(state.numel())]).to(device)
        small_noise = torch.tensor([rng.normalvariate(0, std) for _ in range(state.numel())]).to(device)
        plaintext_weights = state + big_noise
        encrypted_weights = state + small_noise

    #write reply
    #TODO: encryption, ZK proof

    plaintext_weights_record = ArrayRecord(torch_state_dict = {"plaintext-weights": plaintext_weights})
    encrypted_weights = pickle.dumps(encrypted_weights)
    config_rec = ConfigRecord({
        "active" : True, 
        "encrypted-weights": encrypted_weights
        })
    num_examples_record = MetricRecord({"num-examples": len(private_train_loader.dataset)})
    return Message(
        content = RecordDict(records = {
            "plaintext-weights": plaintext_weights_record,
            "config": config_rec,
            "num-examples": num_examples_record
        }), reply_to = msg)

    

@app.evaluate()
def evaluate(msg: Message, context: Context):
    # Load the model and initialize it with the received weights
    model = model_loading.Model()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    model.to(device)
    criterion = model_loading.loss()

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, test_loader = data_loading.load_data(partition_id, num_partitions, batch_size)

    accuracy, loss = util.test(model, criterion, test_loader, device)

    # Construct and return reply Message
    metrics = {
        "eval_acc": accuracy,
        "eval_loss": loss,
        "num-examples": len(test_loader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)