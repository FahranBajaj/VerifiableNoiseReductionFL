from collections.abc import Callable, Iterable
from logging import INFO, DEBUG, WARNING
import random
import math
import gc
import pickle

import numpy as np
import torch
import tenseal as ts

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

from src import feddmc, util, model_loading, ckks, attacks
from src.normality_tests import jarque_bera
import src.config

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
        use_dp: bool = True,
        noise_reduction: bool = True,
        num_updates: int | None = None,
        pca_components: int = 5,
        feddmc_alpha: float = 0.8,
        min_cluster_fraction: float = 0.05,
        jarque_bera_significance: float = 0.05,
        expected_std = 0
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

        if not (num_updates is None or (isinstance(num_updates, int) and num_updates >= 0)):
            raise ValueError("num_updates must be a nonnegative integer (or None)")
        
        if not (fraction_malicious >= 0 and fraction_malicious <= 1):
            raise ValueError("fraction_malicious must be a number between 0 and 1 (inclusive)")
        
        if not (feddmc_alpha >= 0 and feddmc_alpha <= 1):
            raise ValueError("feddmc_alpha must be a number between 0 and 1 (inclusive)")
        
        if not (min_cluster_fraction >= 0 and min_cluster_fraction < 0.5):
            raise ValueError("min_cluster_fraction must be a nonnegative number strictly less than 0.5")
        self.max_num_updates: float = (math.inf if num_updates is None else num_updates)
        if self.max_num_updates == 0:
            self.fraction_train = 0
            self.fraction_evaluate = 0
        self.fraction_malicious: float = fraction_malicious
        self.malicious_ids: list[int] = []
        self.use_dp: bool = use_dp
        self.noise_reduction: bool = noise_reduction
        self.pca_components: int = pca_components
        self.feddmc_alpha: float = feddmc_alpha
        self.min_cluster_fraction: float = min_cluster_fraction
        self.jarque_bera_alpha: float = jarque_bera_significance
        self.expected_std: float = expected_std
        self.trust_scores: dict[int, float] = {}
        self.ids_to_test_rejections: dict[int, int] = {}
        self.max_rejections = math.ceil(3*self.jarque_bera_alpha*self.max_num_updates)
        src.config.max_rejections = self.max_rejections
        self.current_nodes: list[int] = []
        self.trained_this_round: bool = True
        self.ids_to_ciphertexts: dict[int, ts.CKKSVector]
        self.ids_to_num_examples: dict[int, int]
        self.num_total_clients: int
        self.ids_to_num_examples = {}
        self.total_message_size: int = 0

    def configure_train(
            self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
        ) -> Iterable[Message]:
        messages = super().configure_train(server_round, arrays, config, grid)
        if util.read_toml("measure-messages"):
            self.total_message_size += sum([len(pickle.dumps(message)) for message in messages])

        return messages

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        if util.read_toml("measure-messages"):
            self.total_message_size += sum([len(pickle.dumps(message)) for message in replies])
        if not util.read_toml("use-feddmc"):
            #regular FedAvg
            return super().aggregate_train(server_round, replies)
        
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid_replies:
            return None, None
        
        #We receive plaintext model weights to inspect and aggregate
        log(INFO, f"aggregate_train: received {len(replies)} replies, inspecting and aggregating plaintext weights")
        ids_to_plaintext_weights = {}
        plaintext_weights = np.array([])
        active_clients = []
        for reply in replies:
            id = reply.metadata.src_node_id
            records = reply.content
            if not id in self.trust_scores.keys():
                self.trust_scores[id] = 0.5

            active_clients.append(id)
            client_plaintext_weights = records["array"]["plaintext-weights"].numpy()
            self.ids_to_num_examples[id] = records["num-examples"]["num-examples"]
            plaintext_weights = np.append(plaintext_weights, [client_plaintext_weights], axis = 0) if len(plaintext_weights) > 0 else [client_plaintext_weights]
            ids_to_plaintext_weights[id] = client_plaintext_weights

        #FedDMC
        low_dim_weights = feddmc.pca(plaintext_weights, self.pca_components)
        benign_idxs, malicious_idxs = feddmc.benign_and_malicious(low_dim_weights, int(self.min_cluster_fraction * len(active_clients)))

        #Positive indicates that clustering did not fail
        if len(benign_idxs) > 0:
            feddmc.update_trust_scores(
                self.trust_scores, 
                [active_clients[int(index)] for index in benign_idxs], 
                [active_clients[int(index)] for index in malicious_idxs], 
                self.feddmc_alpha)

        aggregated_weights = torch.zeros(plaintext_weights[0].size).to(device)
        total_examples = 0
        for id in active_clients:
            if self.trust_scores[id] >= 0.75:
                total_examples += self.ids_to_num_examples[id]
                aggregated_weights += torch.tensor(ids_to_plaintext_weights[id]).to(device)*self.ids_to_num_examples[id]

        if total_examples == 0:
            #This happens at the beginning since detection threshold is 0.75
            self.ids_to_num_examples = {}
            src.config.total_model_updates += 1 #increase because this round was a dp exposure despite no model update
            src.config.trust_scores = self.trust_scores
            if src.config.total_model_updates == self.max_num_updates:
                self.fraction_train = 0
                self.fraction_evaluate = 0
            src.config.last_update_round = server_round
            return None, None

        aggregated_weights /= total_examples
        self.ids_to_num_examples = {}
        aggregated_weights = ArrayRecord(util.vec_to_state_dict(model_loading.model().state_dict(), aggregated_weights.to(device)))

        # Aggregate custom metrics if aggregation fn was provided
        aggregated_metrics = self.train_metrics_aggr_fn(
                [msg.content for msg in valid_replies],
                self.weighted_by_key,
        )

        src.config.total_model_updates += 1
        src.config.trust_scores = self.trust_scores
        if src.config.total_model_updates == self.max_num_updates:
            self.fraction_train = 0
            self.fraction_evaluate = 0
        src.config.last_update_round = server_round
        gc.collect()
        return aggregated_weights, aggregated_metrics