import math
import random
import os
from logging import ERROR, DEBUG

import torch
import tenseal as ts
from flwr.app import Context, Message, RecordDict, ConfigRecord, MetricRecord, ArrayRecord
from flwr.clientapp import ClientApp
from flwr.common.logger import log
from opacus import PrivacyEngine

from src import model_loading, data_loading, util

os.environ["RAY_memory_monitor_refresh_ms"] = "0"

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

    id = context.node_config["partition-id"]

    if "Malicious" not in context.state.keys():
        log(ERROR, f"No record of whether node {id} is malicious")
        raise RuntimeError(f"No record of whether node {id} is malicious")
    
    if config["Instruction"] == "SENDWEIGHTS":
        if "LocalWeights" not in context.state.keys():
            log(ERROR, f"Node {id} has no saved local model weights")
            raise RuntimeError(f"Node {id} has no saved local model weights")
            
        plaintext_weights_record = context.state["LocalWeights"]
        trivial_metric_record = MetricRecord({"num-examples": 1}) #strategy expects a metric record after every iteration
        del context.state["LocalWeights"]
        config_rec = ConfigRecord({"active" : True})
        return Message(
            content = RecordDict(records = {
                "plaintext-weights": plaintext_weights_record,
                "metric": trivial_metric_record,
                "config": config_rec,
            }), reply_to = msg)

    elif config["Instruction"] == "TRAIN":
        #load data
        num_partitions = context.node_config["num-partitions"]
        batch_size = context.run_config["batch-size"]
        #TODO: implement the below function
        trainloader, _ = data_loading.load_data(id, num_partitions, batch_size)

        #load model
        #TODO: implement below functions (when I have data, decide a model architecture)
        model = model_loading.Model()
        criterion = model_loading.loss()
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

        #train
        private_model.train()
        local_epochs = context.run_config["local-epochs"]
        for _ in range(local_epochs):
            for batch in private_train_loader:
                optimizer.zero_grad()
                criterion(private_model(batch["image"].to(device)), batch["label"].to(device)).backward()
                optimizer.step()

        #add nosie
        state = util.state_dict_to_vec(private_model.state_dict()).to(device)
        plaintext_weights = torch.tensor([]).to(device) #initialize variables for later use
        low_noise_weights = torch.tensor([]).to(device)
        encrypted_differences = torch.tensor([]).to(device)

        #should be equivalent to noise multiplier computed in DP accounting notebook, NOT the "ratio"
        #Potential TODO: compute noise multiplier somewhere in code instead of inputting it in the config file
        noise_multiplier = context.run_config["noise-multiplier"]
        #TODO: make trusted parties a multiplier times the number of total parties? Would need to communicate total parties from server
        trusted_parties = context.run_config["trusted-parties"]
        trusted_multiplier = 1/math.sqrt(trusted_parties-1)

        if context.run_config["reproducible"]:
            std = torch.ones_like(state)*noise_multiplier*learning_rate*clipping_norm*local_epochs/batch_size
            plaintext_weights = state + torch.normal(0, std).to(device)
            low_noise_weights = state + torch.normal(0, std*trusted_multiplier).to(device)
        else:
            rng = random.SystemRandom()
            std = noise_multiplier*learning_rate*clipping_norm*local_epochs/batch_size
            #use threadsafe .normalvariate() instead of .gauss() since we may have multiple clients running at once
            big_noise = torch.tensor([rng.normalvariate(0, std) for _ in range(state.numel())]).to(device)
            small_noise = torch.tensor([rng.normalvariate(0, std*trusted_multiplier) for _ in range(state.numel())]).to(device)
            plaintext_weights = state + big_noise
            low_noise_weights = state + small_noise
        encrypted_differences = len(private_train_loader.dataset)*(low_noise_weights - plaintext_weights)

        #store plaintext weights, write reply
        context.state["LocalWeights"] = ArrayRecord({"plaintext-weights": plaintext_weights})
        encrypted_differences = ts.ckks_vector(ts.context_from(config["CKKS-context"]), encrypted_differences.cpu()).serialize()
        config_rec = ConfigRecord({
            "active" : True, 
            "encrypted-difference": encrypted_differences
            })
        num_examples_record = MetricRecord({"num-examples": len(private_train_loader.dataset)})
        empty_array_rec = ArrayRecord({}) #strategy expects exactly one array record per iteration
        return Message(
            content = RecordDict(records = {
                "config": config_rec,
                "num-examples": num_examples_record,
                "array": empty_array_rec
            }), reply_to = msg)
    
    else:
        log(ERROR, "Unrecognized instruction")
        raise Exception("Unrecognized instruction")

    

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