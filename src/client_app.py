import math
import random
import os
from logging import ERROR, DEBUG
import gc

import torch
import tenseal as ts
from flwr.app import Context, Message, RecordDict, ConfigRecord, MetricRecord, ArrayRecord
from flwr.clientapp import ClientApp
from flwr.common.logger import log
from opacus import PrivacyEngine

from src import model_loading, data_loading, util, attacks
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
    if "Malicious" in config.keys():
        context.state["Malicious"] = ConfigRecord({"Malicious": config["Malicious"]})

        if not config["Active"]:
            return Message(content = RecordDict(configs_records = {"fitres.metrics" : ConfigRecord({"active" : False})}), reply_to = msg)

    id = context.node_config["partition-id"]

    if "Malicious" not in context.state.keys():
        log(ERROR, f"No record of whether node {id} is malicious")
        raise RuntimeError(f"No record of whether node {id} is malicious")
    
    if config["Instruction"] == "SENDWEIGHTS":
        if "NoisyWeights" not in context.state.keys():
            log(ERROR, f"Node {id} has no saved local model weights")
            raise RuntimeError(f"Node {id} has no saved local model weights")
            
        plaintext_weights_record = context.state["NoisyWeights"]
        trivial_metric_record = MetricRecord({"num-examples": 1}) #strategy expects a metric record after every iteration
        del context.state["NoisyWeights"]
        del context.state["RawWeights"]
        config_rec = ConfigRecord({"active" : True})
        return Message(
            content = RecordDict(records = {
                "plaintext-weights": plaintext_weights_record,
                "metric": trivial_metric_record,
                "config": config_rec,
            }), reply_to = msg)

    elif config["Instruction"] == "TRAIN":
        learning_rate = context.run_config["learning-rate"]
        clipping_norm = context.run_config["max-norm"]
        local_epochs = context.run_config["local-epochs"]
        batch_size = context.run_config["batch-size"]
        dataset = Datasets.EMNIST if context.run_config["dataset"] == "EMNIST" else Datasets.WEATHER if context.run_config["dataset"] == "WEATHER" else Datasets.CIFAR10 if context.run_config["dataset"] == "CIFAR10" else Datasets.MNIST
        device = torch.accelerator.current_accelerator().type if (torch.accelerator.is_available() and dataset != Datasets.CIFAR10) else "cpu"
        if not ("RawWeights" in context.state.keys()):
            #need to train and compute local weights
            #load data
            num_partitions = context.node_config["num-partitions"]
            trainloader = data_loading.load_data(id, num_partitions, batch_size)

            #load model
            model = model_loading.model()
            criterion = model_loading.loss()
            model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
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
            context.state["NumExamples"] = MetricRecord({
                #len(private_train_loader.dataset) inaccurate since drop_last = True
                "num-examples": len(private_train_loader)*batch_size
            })

            #train
            if context.state["Malicious"]["Malicious"]:
                context.state["RawWeights"] = attacks.malicious_update(private_model, optimizer, private_train_loader, context)
            else:
                private_model.train()
                for _ in range(local_epochs):
                    for batch in private_train_loader:
                        optimizer.zero_grad()
                        criterion(private_model(batch[util.X_key(dataset)].to(device)), batch[util.y_key(dataset)].to(device)).backward()
                        optimizer.step()

                context.state["RawWeights"] = ArrayRecord({"raw-weights": util.state_dict_to_vec(private_model.state_dict())})

        state = torch.tensor(context.state["RawWeights"]["raw-weights"].numpy()).to(device)
        noise_multiplier = config["noise-multiplier"]
        std = torch.ones_like(state)*noise_multiplier*learning_rate*clipping_norm*local_epochs/batch_size
        plaintext_weights = state + torch.normal(torch.zeros_like(state), std).to(device)
        context.state["NoisyWeights"] = ArrayRecord({"plaintext-weights": plaintext_weights})
        num_examples_record = context.state["NumExamples"]
        empty_array_rec = ArrayRecord({}) #strategy expects exactly one array record per iteration
        config_rec: ConfigRecord
        if context.run_config["noise-reduction"]:
            trusted_parties = config["trusted-parties"]
            trusted_multiplier = 1/math.sqrt(trusted_parties-1)
            #TODO: malicious clients shouldn't add any noise to weights that will actually be aggreagated
            low_noise_weights = state + torch.normal(torch.zeros_like(state), std*trusted_multiplier).to(device)
            encrypted_differences = num_examples_record["num-examples"]*(low_noise_weights - plaintext_weights)

            #store plaintext weights, write reply
            encrypted_differences = ts.ckks_vector(ts.context_from(config["CKKS-context"]), encrypted_differences.cpu()).serialize()
            config_rec = ConfigRecord({
                "active" : True, 
                "encrypted-difference": encrypted_differences
                })
        else:
            config_rec = ConfigRecord({"active" : True})

        return Message(
            content = RecordDict(records = {
                "config": config_rec,
                "num-examples": num_examples_record,
                "array": empty_array_rec
            }), reply_to = msg)
    
    else:
        log(ERROR, "Unrecognized instruction")
        raise Exception("Unrecognized instruction")