from collections.abc import Callable
from flwr.server.strategy.fedavg import FedAvg 
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    ArrayRecord
)
from flwr.common.logger import log
from flwr.compat.common import recorddict_compat

import random
import pickle
import numpy as np
import torch
from logging import WARNING

import feddmc
import util
import model_loading

class ZKFLStrategy(FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        fraction_malicious: float = 0.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: (
            Callable[
                [int, NDArrays, dict[str, Scalar]],
                tuple[float, dict[str, Scalar]] | None,
            ]
            | None
        ) = None,
        on_fit_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        on_evaluate_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        accept_failures: bool = True,
        initial_parameters: Parameters | None = None,
        fit_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        pca_components: int = 5, 
        feddmc_alpha: float = 0.8,
        min_cluster_fraction: float = 0.03
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit, 
            fraction_evaluate=fraction_evaluate, 
            min_fit_clients=min_fit_clients, 
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            inplace=False
        )
        
        if not (fraction_malicious >= 0 and fraction_malicious <= 1):
            raise ValueError("fraction_malicious must be a number between 0 and 1 (inclusive)")
        
        if not (feddmc_alpha >= 0 and feddmc_alpha <= 1):
            raise ValueError("feddmc_alpha must be a number between 0 and 1 (inclusive)")
        
        if not (min_cluster_fraction >= 0 and min_cluster_fraction < 0.5):
            raise ValueError("min_cluster_fraction must be a nonnegative number strictly less than 0.5")
        self.fraction_malicious = fraction_malicious
        self.pca_components = pca_components
        self.alpha = feddmc_alpha
        self.min_cluster_fraction = min_cluster_fraction
        self.trust_scores = {}

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        parameters : Parameters
            The current (global) model parameters.
        client_manager : ClientManager
            The client manager which holds all currently connected clients.

        Returns
        -------
        fit_configuration : List[Tuple[ClientProxy, FitIns]]
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `FitIns` for this particular `ClientProxy`. If a particular `ClientProxy`
            is not included in this list, it means that this `ClientProxy`
            will not participate in the next round of federated learning.
        """

        if server_round == 0:
            #TODO: write a message using flower's build-in log function
            #TODO: need to figure out if the first round is number 0 or 1

            #Below code adapted from Flower's FedAvg strategy
            config = {}
            if self.on_fit_config_fn is not None:
                # Custom fit config function provided
                config = self.on_fit_config_fn(server_round)

            #select all clients
            clients = client_manager.all()
            clients_and_configs = [(client, config.copy()) for client in clients]
            num_available = client_manager.num_available()
            
            #pick clients to be active
            random.shuffle(clients_and_configs)
            num_participating = self.num_fit_clients(num_available)[0]
            for i in range(num_participating):
                clients_and_configs[i][1]["Active"] = True
            for i in range(num_participating, num_available):
                clients_and_configs[i][1]["Active"] = False

            #pick clients to be malicious
            random.shuffle(clients_and_configs)
            num_malicious = round(num_available*self.fraction_malicious)
            for i in range(num_malicious):
                clients_and_configs[i][1]["Malicious"] = True
            for i in range(num_malicious, num_available):
                clients_and_configs[i][1]["Malicious"] = False
            
            # Return client/config pairs
            return [(client, FitIns(parameters, config)) for client, config in clients_and_configs]
        
        return super.configure_fit(server_round, parameters, client_manager)

        

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate training results.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        results : List[Tuple[ClientProxy, FitRes]]
            Successful updates from the previously selected and configured
            clients. Each pair of `(ClientProxy, FitRes)` constitutes a
            successful update from one of the previously selected clients. Note
            that not all previously selected clients are necessarily included in
            this list: a client might drop out and not submit a result. For each
            client that did not submit an update, there should be an `Exception`
            in `failures`.
        failures : List[Union[Tuple[ClientProxy, FitRes], BaseException]]
            Exceptions that occurred while the server was waiting for client
            updates.

        Returns
        -------
        parameters : Tuple[Optional[Parameters], Dict[str, Scalar]]
            If parameters are returned, then the server will treat these as the
            new global model parameters (i.e., it will replace the previous
            parameters with the ones returned from this method). If `None` is
            returned (e.g., because there were only failures and no viable
            results) then the server will no update the previous model
            parameters, the updates received in this round are discarded, and
            the global model parameters remain the same.
        """
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        active_clients = []
        plaintext_weights = np.array([])
        ids_to_encrypted_weights = {}
        for client_proxy, fit_res in results:
            id = client_proxy.node_id
            if not id in self.trust_scores.keys():
                self.trust_scores[client_proxy.node_id] = 0.5
            message_payload = fit_res.metrics
            if message_payload["active"]:
                active_clients.append(id)
                #deserialize client's model weights
                client_plaintext_weights = pickle.loads(message_payload["plaintext-weights"])
                plaintext_weights = np.append(plaintext_weights, [client_plaintext_weights], axis = 0) if len(plaintext_weights) > 0 else [client_plaintext_weights]

        #FedDMC
        low_dim_weights = feddmc.pca(plaintext_weights, self.pca_components)
        benign_idxs, malicious_idxs = feddmc.benign_and_malicious(low_dim_weights, int(self.min_cluster_fraction * len(active_clients)))

        #Positive indicates that clustering did not fail
        if len(benign_idxs) > 0:
            feddmc.update_trust_scores(
                self.trust_scores, 
                [active_clients[int(index)] for index in benign_idxs], 
                [active_clients[int(index)] for index in malicious_idxs], 
                self.alpha)

        #TODO: verify zk proofs

        kept_results = []
        for client_proxy, fit_res in results:
            id = client_proxy.node_id
            if id in active_clients and self.trust_scores[id] >= 0.5:
                kept_results.append(fit_res)

        #Average results from benign clients
        #TODO: change to secure aggregation
        aggregated_weights = torch.zeros_like(plaintext_weights[0])
        total_examples = 0
        for fit_res in kept_results:
                encrypted_weights = pickle.loads(fit_res.metrics["encrypted-weights"])
                total_examples += fit_res.num_examples
                aggregated_weights += fit_res.num_examples * encrypted_weights
            
        aggregated_weights /= total_examples
        aggregated_weights = ArrayRecord(util.vec_to_state_dict(model_loading.Model().state_dict(), aggregated_weights))


        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for res in kept_results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return util.arrayrecord_to_parameters(aggregated_weights), metrics_aggregated