import pickle
from collections.abc import Callable, Iterable
from logging import INFO

import numpy as np
import torch

from flwr.serverapp import Grid
from flwr.serverapp.strategy.fedavg import FedAvg
from flwr.serverapp.strategy import strategy_utils
from flwr.app import (
    Message,
    MessageType,
    RecordDict,
    MetricRecord,
    ArrayRecord,
    ConfigRecord
)
from flwr.common.logger import log

from src import feddmc, util, model_loading

class ZKFLStrategy(FedAvg):
    def __init__(
        self,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        fraction_malicious: float = 0.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        weighted_by_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
        train_metrics_aggr_fn: (
            Callable[[list[RecordDict], str], MetricRecord] | None
        ) = None,
        evaluate_metrics_aggr_fn: (
            Callable[[list[RecordDict], str], MetricRecord] | None
        ) = None,
        pca_components: int = 5,
        feddmc_alpha: float = 0.8,
        min_cluster_fraction: float = 0.03
    ) -> None:
        super().__init__(
            fraction_train, 
            fraction_evaluate,
            min_train_nodes,
            min_evaluate_nodes,
            min_available_nodes,
            weighted_by_key,
            arrayrecord_key,
            configrecord_key,
            train_metrics_aggr_fn,
            evaluate_metrics_aggr_fn
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

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        if server_round == 0:
            log(INFO, "configure_train: configuring first round of training")
            #TODO: need to figure out if the first round is number 0 or 1

            #select all clients
            all_ids = grid.get_node_ids()
            ids_and_configs = [(id, config.copy()) for id in all_ids]
            total_nodes = len(ids_and_configs)
            
            #pick clients to be active and malicious
            active_ids, all_ids = strategy_utils.sample_nodes(grid, self.min_available_nodes, max(self.min_train_nodes, int(total_nodes*self.fraction_train)))
            malicious_ids, all_ids = strategy_utils.sample_nodes(grid, 0, int(total_nodes*self.fraction_malicious))
            for id, conf in ids_and_configs:
                conf["Active"] = (id in active_ids)
                conf["Malicious"] = (id in malicious_ids)
            
            # Return messages
            return [Message(RecordDict({
                        self.arrayrecord_key: arrays, 
                        self.configrecord_key: conf
                    }), id, MessageType.TRAIN) for id, conf in ids_and_configs]
        
        return super.configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid_replies:
            return None, None
        
        active_clients = []
        plaintext_weights = np.array([])
        ids_to_ciphertexts = {}
        for reply in replies:
            id = reply.reply_to.dst_node_id
            records = reply.content.records
            if not id in self.trust_scores.keys():
                self.trust_scores[id] = 0.5
            if records["config"]["active"]:
                active_clients.append(id)
                client_plaintext_weights = records["plaintext-weights"]["plaintext-weights"]
                plaintext_weights = np.append(plaintext_weights, [client_plaintext_weights], axis = 0) if len(plaintext_weights) > 0 else [client_plaintext_weights]
                ids_to_ciphertexts[id] = (next(iter(records.metric_records.values()))[self.weighted_by_key], pickle.loads(records["config"]["encrypted-weights"]))

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
        aggregated_weights = torch.zeros_like(plaintext_weights[0])
        total_examples = 0
        for id in active_clients:
            if self.trust_scores[id] >= 0.5:
                num_examples = ids_to_ciphertexts[id][0]
                weights = ids_to_ciphertexts[id][1  ]
                total_examples += num_examples
                aggregated_weights += weights*num_examples
            
        aggregated_weights /= total_examples
        aggregated_weights = ArrayRecord(util.vec_to_state_dict(model_loading.Model().state_dict(), aggregated_weights))


        # Aggregate custom metrics if aggregation fn was provided
        aggregated_metrics = self.train_metrics_aggr_fn(
                [msg.content for msg in valid_replies],
                self.weighted_by_key,
        )

        return aggregated_weights, aggregated_metrics