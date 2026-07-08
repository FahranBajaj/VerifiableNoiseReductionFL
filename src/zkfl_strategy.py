from collections.abc import Callable, Iterable
from logging import INFO, DEBUG, WARNING
import random

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

from src import feddmc, util, model_loading, ckks
from src.anderson_darling import anderson_darling, CRITICAL_THRESHOLDS

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
        min_cluster_fraction: float = 0.03,
        anderson_darling_significance: float = 0.05,
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
        
        if not (fraction_malicious >= 0 and fraction_malicious <= 1):
            raise ValueError("fraction_malicious must be a number between 0 and 1 (inclusive)")
        
        if not (feddmc_alpha >= 0 and feddmc_alpha <= 1):
            raise ValueError("feddmc_alpha must be a number between 0 and 1 (inclusive)")
        
        if not (min_cluster_fraction >= 0 and min_cluster_fraction < 0.5):
            raise ValueError("min_cluster_fraction must be a nonnegative number strictly less than 0.5")
        if not (anderson_darling_significance in CRITICAL_THRESHOLDS.keys()):
            raise ValueError("Anderson-Darling significance level must be one of 0.01, 0.025, 0.05, 0.1, or 0.15")
        self.fraction_malicious: float = fraction_malicious
        self.pca_components: int = pca_components
        self.feddmc_alpha: float = feddmc_alpha
        self.min_cluster_fraction: float = min_cluster_fraction
        self.anderson_darling_alpha: float = anderson_darling_significance
        self.expected_std: float = expected_std
        self.trust_scores: dict[int, float] = {}
        self.num_model_updates: int = 0
        self.current_nodes: list[int] = []
        self.trained_this_round: bool = True
        self.ids_to_ciphertexts: dict[int, ts.CKKSVector]
        self.ids_to_num_examples: dict[int, int]

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        if server_round == 1:
            log(INFO, "configure_train: configuring first round of training")
            self.trained_this_round = True
            self.current_ckks_context = ckks.generage_ckks_context()
            public_context = self.current_ckks_context.copy()
            public_context.make_context_public()
            public_context = public_context.serialize()

            #select all clients
            all_ids = grid.get_node_ids()
            config["Instruction"] = "TRAIN"
            config["CKKS-context"] = public_context
            config["server-round"] = server_round
            ids_and_configs = [(id, config.copy()) for id in all_ids]
            total_nodes = len(ids_and_configs)
            
            #pick clients to be active and malicious
            self.current_nodes, all_ids = strategy_utils.sample_nodes(grid, self.min_available_nodes, max(self.min_train_nodes, int(total_nodes*self.fraction_train)))
            malicious_ids, all_ids = strategy_utils.sample_nodes(grid, 0, int(total_nodes*self.fraction_malicious))
            for id, conf in ids_and_configs:
                conf["Active"] = (id in self.current_nodes)
                conf["Malicious"] = (id in malicious_ids)
            log(
                INFO,
                "configure_train: Sampled %s nodes (out of %s)",
                len(self.current_nodes),
                len(all_ids),
            )
            log(
                INFO,
                "configure_train: Sampled %s malicious nodes (out of %s)",
                len(malicious_ids),
                len(all_ids),
            )
            
            # Return messages
            return [Message(RecordDict({
                        self.arrayrecord_key: arrays, 
                        self.configrecord_key: conf
                    }), id, MessageType.TRAIN) for id, conf in ids_and_configs]
        
        if not self.current_nodes:
            #Select some nodes and have them train
            log(INFO, "configure_train: selecting nodes to train")
            self.trained_this_round = True
            self.current_ckks_context = ckks.generage_ckks_context() #need new context, releasing CKKS decryptions can leak secret key
            public_context = self.current_ckks_context.copy()
            public_context.make_context_public()
            public_context = public_context.serialize()

            if self.fraction_train == 0.0:
                log(WARNING, "configure_train: fraction_train is 0 so no nodes were selected")
                return []
            
            #select nodes
            num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
            sample_size = max(num_nodes, self.min_train_nodes)
            self.current_nodes, all_ids = strategy_utils.sample_nodes(grid, self.min_available_nodes, sample_size)
            log(
                INFO,
                "configure_train: Sampled %s nodes (out of %s)",
                len(self.current_nodes),
                len(all_ids),
            )

            config["Instruction"] == "TRAIN"
            config["CKKS-context"] = public_context
            config["server-round"] = server_round
            self.trained_this_round = True
        else:
            #Tell nodes from last round to send their weights
            log(INFO, "configure_train: gathering plaintext weights from previously-sampled nodes")
            self.trained_this_round = False
            config["Instruction"] == "SENDWEIGHTS"
            config["server-round"] = server_round
            self.trained_this_round = False

        record = RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: config})
        return self._construct_messages(record, self.current_nodes, MessageType.TRAIN)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid_replies:
            return None, None
        
        if self.trained_this_round:
            #We receive ciphertexts 
            #Either aggregate or inspect
            inspecting = random.randint(0, 1)
            self.ids_to_ciphertexts = {}
            self.ids_to_num_examples = {}
            self.total_examples = 0
            for reply in replies:
                id = reply.metadata.src_node_id
                records = reply.content
                if not id in self.trust_scores.keys():
                    self.trust_scores[id] = 0.5
                if records["config"]["active"]:
                    serialized_ciphertet = records["config"]["encrypted-difference"]
                    num_examples = records["num-examples"]["num-examples"]
                    if inspecting:
                        difference = ts.ckks_vector_from(self.current_ckks_context, serialized_ciphertet).decrypt()
                        new_trust_score = 1 - anderson_darling(difference, 0, self.expected_std*num_examples, self.anderson_darling_alpha)
                        self.trust_scores[id] = self.trust_scores[id] * self.feddmc_alpha + (1-self.feddmc_alpha)*new_trust_score
                    else:
                        #keep ciphertexts for aggregation and decryption next round
                        #it would be nice to decrypt now, but we don't know which clients 
                        #we'll include in aggregation until we inspect model weights
                        self.ids_to_ciphertexts[id] = serialized_ciphertet
                        self.ids_to_num_examples[id] = num_examples
                        self.total_examples += num_examples

            if inspecting:
                self.current_nodes = [] #clear out list to indicate we select new clients next iteration
                
            return None, None #no global update was performed
                       
        else:
            #We receive plaintext model weights to inspect and aggregate
            active_clients = []
            ids_to_plaintext_weights = {}
            plaintext_weights = np.array([])
            for reply in replies:
                id = reply.metadata.src_node_id
                records = reply.content
                if not id in self.trust_scores.keys():
                    self.trust_scores[id] = 0.5
                if records["config"]["active"]:
                    active_clients.append(id)
                    client_plaintext_weights = records["plaintext-weights"]["plaintext-weights"].numpy()
                    plaintext_weights = np.append(plaintext_weights, [client_plaintext_weights], axis = 0) if len(plaintext_weights) > 0 else [client_plaintext_weights]
                    ids_to_plaintext_weights[id] = plaintext_weights

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
            aggregated_differneces = [0] * plaintext_weights[0].size
            for id in active_clients:
                if self.trust_scores[id] >= 0.5:
                    aggregated_weights += ids_to_plaintext_weights[id]*self.ids_to_num_examples[id]
                    aggregated_differneces: ts.CKKSVector = aggregated_differneces + ts.ckks_vector_from(self.current_ckks_context, self.ids_to_ciphertexts[id])
                
            aggregated_weights = (aggregated_weights + torch.tensor(aggregated_differneces.decrypt()).to(device)) / self.total_examples
            self.ids_to_ciphertexts = {}
            self.ids_to_num_examples = {}
            aggregated_weights = ArrayRecord(util.vec_to_state_dict(model_loading.Model().state_dict(), torch.tensor(aggregated_weights)).to(device))

            # Aggregate custom metrics if aggregation fn was provided
            aggregated_metrics = self.train_metrics_aggr_fn(
                    [msg.content for msg in valid_replies],
                    self.weighted_by_key,
            )

            self.total_examples = 0
            self.ids_to_num_examples = {}
            self.current_nodes = [] #clear out list to indicate we select new clients next iteration
            return aggregated_weights, aggregated_metrics