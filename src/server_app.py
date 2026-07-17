import math
import csv
import os

import torch
import dp_accounting
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from src import model_loading, data_loading, util
from src.zkfl_strategy import ZKFLStrategy
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
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    fraction_malicious: float = context.run_config["fraction-malicious"]
    max_num_rounds: int = context.run_config["max-num-server-rounds"]
    num_model_updates: int | None = context.run_config["num-model-updates"]
    if num_model_updates < 0:
        num_model_updates = None
    num_trusted_parties: int = max(2, context.run_config["trusted-fraction"] * len(grid.get_node_ids()))
    noise_multiplier: float = compute_noise_multiplier(
        num_trusted_parties,
        context.run_config["epsilon"],
        context.run_config["delta"],
        num_model_updates if num_model_updates is not None else max_num_rounds
    )
    msg_to_clients = ConfigRecord({
        "noise-multiplier": noise_multiplier,
        "trusted-parties": num_trusted_parties
    })

    global WRITE_RESULTS_TO_FILE
    global FILE_TO_WRITE
    WRITE_RESULTS_TO_FILE = context.run_config["write-results"]
    FILE_TO_WRITE = context.run_config["results-directory"] + f"/{context.run_id}results.csv"

    # Load global model
    global_model = model_loading.Model()
    arrays = ArrayRecord(global_model.state_dict())

    expected_std = noise_multiplier*context.run_config["learning-rate"]*context.run_config["max-norm"]*context.run_config["local-epochs"]*math.sqrt(1+(1/(num_trusted_parties - 1)))/context.run_config["batch-size"]
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
        model = model_loading.Model()
        model.load_state_dict(arrays.to_torch_state_dict())
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        model.to(device)
        criterion = model_loading.loss()

        # Load entire test set
        test_loader = data_loading.load_centralized_dataset()

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