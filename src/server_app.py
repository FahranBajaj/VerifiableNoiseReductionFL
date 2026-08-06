import math
import csv
import os
from logging import ERROR, DEBUG
import tomllib
from datetime import datetime
import time

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
ATTACK_TYPES: list[str] = [
    "LABELFLIP",
    "GAUSSIAN",
    "LIT",
    "SCALING",
    "ADAPTIVE"
]

# Create ServerApp
app = ServerApp()

def compute_noise_multiplier(trusted_parties: int,
                             target_epsilon: float,
                             target_delta: float,
                             global_model_updates: int,
                             noise_reduction: bool = True) -> float:
    
    def create_mechanism(noise_multiplier):
        if noise_reduction:
            gaussians=dp_accounting.dp_event.ComposedDpEvent([
                dp_accounting.dp_event.GaussianDpEvent(noise_multiplier=noise_multiplier),
                dp_accounting.dp_event.GaussianDpEvent(noise_multiplier=math.sqrt(trusted_parties)*noise_multiplier/(math.sqrt(trusted_parties-1)))
            ])
        else:
            gaussians = dp_accounting.dp_event.GaussianDpEvent(noise_multiplier=noise_multiplier)
        
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
    dataset = Datasets.EMNIST if dataset == "EMNIST" else Datasets.WEATHER if dataset == "WEATHER" else Datasets.CIFAR10 if dataset == "CIFAR10" else Datasets.MNIST
    attack_type: str = context.run_config["attack-type"]
    adaptive_lambda: float = context.run_config["adaptive-attack-lambda"]
    lit_alpha: float = context.run_config["lit-attack-alpha"]
    use_dp: bool = context.run_config["use-dp"]
    noise_reduction: bool = context.run_config["noise-reduction"]
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

    if fraction_malicious != 0 and attack_type not in ATTACK_TYPES:
        raise ValueError("Nonzero fraction malicious but attack type missing or unknown")
    if fraction_malicious == 0 and attack_type != "":
        raise ValueError("Attack type specified but fraction malicious is zero")

    #compute trusted parties, noise multiplier
    num_trusted_parties: int = int(max(2, trusted_fraction * num_clients))
    noise_multiplier: float = int(use_dp) * compute_noise_multiplier(
        num_trusted_parties,
        epsilon,
        delta,
        num_model_updates if num_model_updates is not None else max_num_rounds,
        noise_reduction
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
                "start-time",
                "dataset",
                "attack-type",
                "adaptive-attack-lambda",
                "lit-attack-alpha",
                "use-dp",
                "noise-reduction",
                "concentration-parameter",
                "num-clients",
                "fraction-malicious",
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
                "start-time": datetime.now(),
                "dataset": dataset.value,
                "attack-type": attack_type,
                "adaptive-attack-lambda": adaptive_lambda if attack_type == "ADAPTIVE" else None,
                "lit-attack-alpha": lit_alpha if attack_type == "LIT" else None,
                "use-dp": use_dp,
                "noise-reduction": noise_reduction,
                "concentration-parameter": concentration_parameter,
                "num-clients": num_clients,
                "fraction-malicious": fraction_malicious,
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

    
    strategy: ZKFLStrategy = ZKFLStrategy(fraction_evaluate = fraction_evaluate, fraction_malicious = fraction_malicious, use_dp = use_dp, noise_reduction = noise_reduction, num_updates = num_model_updates)
    
    start_time = time.perf_counter()
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=msg_to_clients,
        num_rounds=max_num_rounds,
        evaluate_fn=global_evaluate,
    )
    time_elapsed = time.perf_counter() - start_time
    if util.read_toml("write-time"):
        with open("times.txt", 'a') as f:
            f.write(str(time_elapsed) + '\n')
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
        dataset = util.read_toml("dataset")
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() and dataset != Datasets.CIFAR10 else "cpu"
        evaluate_train = util.read_toml("evaluate-train")
        model.to(device)
        criterion = model_loading.loss(train = False)

        test_loader = data_loading.load_full_dataset(test = True)
        test_accuracy, test_loss = util.test(model, criterion, test_loader, device)
        if evaluate_train:
            train_loader = data_loading.load_full_dataset(test = False)
            train_accuracy, train_loss = util.test(model, criterion, train_loader, device)

        fieldnames = ["global-update-round", "test-loss", "test-accuracy"]
        row = {
                "global-update-round": src.config.total_model_updates,
                "test-loss": test_loss,
                "test-accuracy": test_accuracy
            }
        if evaluate_train:
            fieldnames += ["train-loss", "train-accuracy"]
            row["train-loss"] = train_loss
            row["train-accuracy"] = train_accuracy

        if util.read_toml("fraction-malicious") > 0:
            fieldnames += ["detection-accuracy", "detection-precision", "detection-recall"]

        #Detection accuracy, precision, recall
        if len(src.config.malicious_ids) > 0:
            malicious_ids = src.config.malicious_ids
            trust_scores = src.config.trust_scores
            true_positives = sum([(id in trust_scores.keys() and trust_scores[id] < 0.75) for id in malicious_ids])
            false_negatives = sum([id in trust_scores.keys() for id in malicious_ids]) - true_positives
            true_negatives = sum([(trust_scores[id] >= 0.75 and id not in malicious_ids) for id in trust_scores.keys()])
            false_positives = len(trust_scores.keys()) - true_positives - false_negatives - true_negatives
            detection_accuracy = (true_positives + true_negatives)/(true_positives + false_negatives + true_negatives + false_positives)
            detection_precision = 0 if true_positives + false_positives == 0 else true_positives/(true_positives + false_positives)
            detection_recall = true_positives/(true_positives + false_negatives)
            row["detection-accuracy"] = detection_accuracy
            row["detection-precision"] = detection_precision
            row["detection-recall"] = detection_recall

        #Backdoor accuracy
        if util.read_toml("attack-type") in ["LIT", "SCALING"]:
            attack_success_rate, _ = util.test(model, criterion, test_loader, device, backdoor = True)
            fieldnames += ["attack-success-rate"]
            row["attack-success-rate"] = attack_success_rate

        #write results to file
        global WRITE_RESULTS_TO_FILE
        if WRITE_RESULTS_TO_FILE:
            global FILE_TO_WRITE
            with open(FILE_TO_WRITE, 'a', newline = '') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
                if os.path.getsize(FILE_TO_WRITE) == 0:
                    writer.writeheader()

                writer.writerow(row)

        # Return the evaluation metrics
        #del row["global-update-round"]
        return MetricRecord(row)
    
    return None