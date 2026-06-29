"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import Context, Message, RecordDict, ConfigRecord
from flwr.clientapp import ClientApp
import model_loading
import data_loading
import math


# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    if context.run_config["reproducible"]:
        torch.manual_seed(42)

    #Check message to see if this is the first round
    config = msg.content["config"]
    if "Malicious" in config.keys():
        if config["Malicious"]:
            context.state["Malicious"] = True

        if !config["Active"]:
            return Message(content = RecordDict({"config": ConfigRecord({"Active" : False})}), reply_to = msg)

    #TODO: if not and context.state doesn't say whether we're malicious or honest
        #then log a warning and return nothing or fail or something

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

    optimizer = torch.optim.SGD(model.parameters(), lr = context.run_config["learning-rate"])

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
    for _ in local_epochs:
        for batch in private_train_loader:
            optimizer.zero_grad()
            criterion(batch["samples"].to(device), batch["labels"].to(device)).backward()
            optimizer.step()

    #add nosie
    state = private_model.state_dict()
    #Potential TODO: Change these data structures to make them easier to deal with for encryption, inspection, aggregation
    plaintext_state = OrderedDict.fromkeys(state.keys())
    encrypted_state = OrderedDict.fromkeys(state.keys())

    #should be equivalent to noise multiplier computed in DP accounting notebook, NOT the "ratio"
    #Potential TODO: compute noise multiplier somewhere in code instead of inputting it in the config file
    noise_multiplier = context.run_config["noise-multiplier"]
    #TODO: configure number of trusted parties
    trusted_parties = msg.content["config"]["trusted-parties"]
    trusted_multiplier = 1/math.sqrt(trusted_parties-1)

    if context.run_config["reproducible"]:
        for name, tensor in state.items():
            std = torch.ones_like(tensor)*noise_multiplier*learning_rate*clipping_norm*local_epochs/batch_size
            plaintext_state[name] = tensor + torch.normal(0, std).to(device)
            encrypted_state[name] = tensor + torch.normal(0, std*trusted_multiplier).to(device)
    else:
        rng = random.SystemRandom()
        for name, tensor in state.items():
            std = noise_multiplier*learning_rate*clipping_norm*local_epochs/batch_size
            big_noise = torch.tensor([rng.gauss(0, std) for _ in range(tensor.numel())]).reshape(tensor.shape).to(device)
            small_noise = torch.tensor([rng.gauss(0, std*trusted_multiplier) for _ in range(tensor.numel())]).reshape(tensor.shape).to(device)
            plaintext_state[name] = tensor + big_noise
            encrypted_state[name] = tensor + small_noise

    #write reply

    raise NotImplementedError()

@app.evaluate()
def evaluate(msg: Message, context: Context):
    raise NotImplementedError()