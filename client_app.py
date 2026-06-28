"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import Context, Message, RecordDict, ConfigRecord
from flwr.clientapp import ClientApp
import model_loading
import data_loading


# Flower ClientApp
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
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

    privacy_engine = PrivacyEngine(secure_mode=True)
    private_model, optimizer, private_train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=trainloader,
        noise_multiplier=context.run_config["noise-multiplier"],
        max_grad_norm=context.run_config["max-norm"],
        poisson_sampling = False
    )

    criterion = model_loading.loss()

    #train
    model.train()
    for _ in context.run_config["local-epochs"]:
        for batch in loader:
            optimizer.zero_grad()
            criterion(batch["samples"].to(device), batch["labels"].to(device)).backward()
            optimizer.step()


    #add nosie

    #write reply

    raise NotImplementedError()

@app.evaluate()
def evaluate(msg: Message, context: Context):
    raise NotImplementedError()