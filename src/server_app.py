import math

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from src import model_loading, data_loading, util
from src.zkfl_strategy import ZKFLStrategy

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    fraction_malicious: float = context.run_config["fraction-malicious"]
    num_rounds: int = context.run_config["num-server-rounds"]

    # Load global model
    global_model = model_loading.Model()
    arrays = ArrayRecord(global_model.state_dict())

    expected_std = context.run_config["noise-multiplier"]*context.run_config["learning-rate"]*context.run_config["max-norm"]*context.run_config["local-epochs"]*math.sqrt(1+(1/(context.run_config["trusted-parties"] - 1)))/context.run_config["batch-size"]
    strategy: ZKFLStrategy = ZKFLStrategy(fraction_evaluate = fraction_evaluate, fraction_malicious = fraction_malicious, expected_std = expected_std)

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=None,
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    if context.run_config["save-model"]:
        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

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

    # Return the evaluation metrics
    return MetricRecord({"accuracy": accuracy, "loss": loss})
