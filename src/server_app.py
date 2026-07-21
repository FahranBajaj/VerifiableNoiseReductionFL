import math
import csv
import os
from logging import ERROR, DEBUG
import tomllib

import torch
import dp_accounting
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.common.logger import log

from src import model_loading, data_loading, util
from src.zkfl_strategy import ZKFLStrategy
from src.util import Datasets
import src.config

WRITE_RESULTS_TO_FILE: bool
FILE_TO_WRITE: str

# Create ServerApp
app = ServerApp()

def compute_noise_multiplier(trusted_parties: int,
                             target_epsilon: float,
                             target_delta: float,
                             global_model_updates: int) -> float:
    
    def create_mechanism(noise_multiplier):
        gaussians=dp_accounting.dp_event.ComposedDpEvent([
            dp_accounting.dp_event.GaussianDpEvent(noise_multiplier=noise_multiplier),
            dp_accounting.dp_event.GaussianDpEvent(noise_multiplier=math.sqrt(trusted_parties)*noise_multiplier/(math.sqrt(trusted_parties-1)))
        ])
        
        full_mechanism = dp_accounting.dp_event.SelfComposedDpEvent(event = gaussians, count = int(global_model_updates))
        return full_mechanism

    required_noise_multiplier = dp_accounting.mechanism_calibration.calibrate_dp_mechanism(
        dp_accounting.pld.PLDAccountant, 
        create_mechanism,
        target_epsilon,
        target_delta,
        dp_accounting.mechanism_calibration.ExplicitBracketInterval(1, 200)) #200 is a loose upper bound for all the tests I'm running
    
    return required_noise_multiplier

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    dataset: str = context.run_config["dataset"]
    dataset = Datasets.WEATHER if dataset == "WEATHER" else Datasets.CIFAR10 if dataset == "CIFAR10" else Datasets.MNIST
    concentration_parameter = context.run_config["concentration-parameter"]
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    fraction_malicious: float = context.run_config["fraction-malicious"]
    max_num_rounds: int = context.run_config["max-num-server-rounds"]
    num_model_updates: int | None = context.run_config["num-model-updates"]
    if num_model_updates < 0:
        num_model_updates = None
    learning_rate: float = context.run_config["learning-rate"]
    clipping_norm: float = context.run_config["max-norm"]
    local_epochs: float = context.run_config["local-epochs"]
    batch_size: int = context.run_config["batch-size"]
    trusted_fraction: float = context.run_config["trusted-fraction"]
    epsilon: float = context.run_config["epsilon"]
    delta: float= context.run_config["delta"]
    id: int = context.run_id
    global WRITE_RESULTS_TO_FILE
    global FILE_TO_WRITE
    WRITE_RESULTS_TO_FILE = context.run_config["write-results"]
    FILE_TO_WRITE = context.run_config["results-directory"] + f"/{id}results.csv"
    num_clients = len(grid.get_node_ids())

    #compute trusted parties, noise multiplier
    num_trusted_parties: int = max(2, trusted_fraction * num_clients)
    noise_multiplier: float = compute_noise_multiplier(
        num_trusted_parties,
        epsilon,
        delta,
        num_model_updates if num_model_updates is not None else max_num_rounds
    )
    msg_to_clients = ConfigRecord({
        "noise-multiplier": noise_multiplier,
        "trusted-parties": num_trusted_parties
    })

    #write to runinfo csv
    if WRITE_RESULTS_TO_FILE:
        with open("runinfo.csv", 'a', newline = '') as csvfile:
            fieldnames = [
                "id",
                "dataset",
                "concentration-parameter",
                "num-clients",
                "fraction-malicious",
                "attack-type",
                "local-epochs",
                "learning-rate",
                "batch-size",
                "trusted-fraction",
                "epsilon",
                "delta",
                "clipping-norm",
                "noise-multiplier"
            ]
            writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
            writer.writerow({
                "id": id,
                "dataset": dataset.value,
                "concentration-parameter": concentration_parameter,
                "num-clients": num_clients,
                "fraction-malicious": fraction_malicious,
                "attack-type": None,
                "local-epochs": local_epochs,
                "learning-rate": learning_rate,
                "batch-size": batch_size,
                "trusted-fraction": trusted_fraction,
                "epsilon": epsilon,
                "delta": delta,
                "clipping-norm": clipping_norm,
                "noise-multiplier": noise_multiplier
            })

    # Load global model
    global_model = model_loading.model()
    arrays = ArrayRecord(global_model.state_dict())

    expected_std = noise_multiplier*learning_rate*clipping_norm*local_epochs*math.sqrt(1+(1/(num_trusted_parties - 1)))/batch_size
    strategy: ZKFLStrategy = ZKFLStrategy(fraction_evaluate = fraction_evaluate, fraction_malicious = fraction_malicious, num_updates = num_model_updates, expected_std = expected_std)

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=msg_to_clients,
        num_rounds=max_num_rounds,
        evaluate_fn=global_evaluate,
    )

    if context.run_config["save-model"]:
        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord | None:
    """Evaluate model on central data."""

    if server_round == src.config.last_update_round:
        # Load the model and initialize it with the received weights
        model = model_loading.model()
        model.load_state_dict(arrays.to_torch_state_dict())
        with open("pyproject.toml", 'rb') as f:
            config_dict = tomllib.load(f)["tool"]["flwr"]["app"]["config"]

        dataset = Datasets.WEATHER if config_dict["dataset"] == "WEATHER" else Datasets.CIFAR10 if config_dict["dataset"] == "CIFAR10" else Datasets.MNIST
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() and dataset != Datasets.CIFAR10 else "cpu"
        model.to(device)
        criterion = model_loading.loss(train = False)

        # Load entire test set
        test_loader = data_loading.load_test_dataset()

        # Evaluate the global model on the test set
        accuracy, loss = util.test(model, criterion, test_loader, device)

        #write results to file
        global WRITE_RESULTS_TO_FILE
        if WRITE_RESULTS_TO_FILE:
            global FILE_TO_WRITE
            with open(FILE_TO_WRITE, 'a', newline = '') as csvfile:
                fieldnames = ["global_update_round", "loss", "accuracy"]
                writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
                if os.path.getsize(FILE_TO_WRITE) == 0:
                    writer.writeheader()
                    
                writer.writerow({
                    "global_update_round": src.config.total_model_updates,
                    "loss": loss,
                    "accuracy": accuracy
                })

        # Return the evaluation metrics
        return MetricRecord({"accuracy": accuracy, "loss": loss})
    
    return None