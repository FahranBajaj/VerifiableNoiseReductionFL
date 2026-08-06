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
        self.total_message_size: int = 0

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        if server_round == 1 and self.fraction_train > 0:
            log(INFO, "configure_train: configuring first round of training")
            self.trained_this_round = True
            self.current_ckks_context = ckks.generate_ckks_context()
            public_context = self.current_ckks_context.copy()
            public_context.make_context_public()
            public_context = public_context.serialize()

            #select all clients
            all_ids = grid.get_node_ids()
            self.num_total_clients = len(all_ids)
            config["Instruction"] = "TRAIN"
            config["CKKS-context"] = public_context
            config["server-round"] = server_round
            
            #pick clients to be active and malicious
            self.current_nodes, all_ids = strategy_utils.sample_nodes(grid, self.min_available_nodes, max(self.min_train_nodes, int(self.num_total_clients*self.fraction_train)))
            self.malicious_ids, all_ids = strategy_utils.sample_nodes(grid, 0, round(self.num_total_clients*self.fraction_malicious))

            #LIT attack
            if util.read_toml("attack-type") == "LIT":
                self.node_ids_to_partition_ids: dict[int,int] = {}
                record = RecordDict({"config": ConfigRecord({"Instruction": "SENDID"})})
                messages = self._construct_messages(record, all_ids, MessageType.TRAIN)
                replies = grid.send_and_receive(messages)
                for msg in replies:
                    self.node_ids_to_partition_ids[msg.metadata.src_node_id] = msg.content["config"]["partition-id"]

                log(INFO, "configure_train: Computing LIT attack update...")
                lit_update = attacks.lit_attack_update(self.malicious_ids, arrays, self.num_total_clients, self.node_ids_to_partition_ids)
                config["LIT-update"] = pickle.dumps(lit_update)

            ids_and_configs = [(id, config.copy()) for id in all_ids]
            
            src.config.malicious_ids = self.malicious_ids
            for id, conf in ids_and_configs:
                conf["Active"] = (id in self.current_nodes)
                conf["Malicious"] = (id in self.malicious_ids)
            log(
                INFO,
                "configure_train: Sampled %s nodes (out of %s)",
                len(self.current_nodes),
                len(all_ids),
            )
            log(
                INFO,
                "configure_train: Sampled %s malicious nodes (out of %s)",
                len(self.malicious_ids),
                len(all_ids),
            )
            
            # Return messages
            messages =  [Message(RecordDict({
                        self.arrayrecord_key: arrays, 
                        self.configrecord_key: conf
                    }), id, MessageType.TRAIN) for id, conf in ids_and_configs]

            if util.read_toml("measure-messages"):
                self.total_message_size += sum([len(pickle.dumps(message)) for message in messages])
            return messages
        
        if not self.current_nodes:
            #Select some nodes and have them train
            log(INFO, "configure_train: selecting nodes to train")
            self.trained_this_round = True
            self.current_ckks_context = ckks.generate_ckks_context() #need new context, releasing CKKS decryptions can leak secret key
            public_context = self.current_ckks_context.copy()
            public_context.make_context_public()
            public_context = public_context.serialize()

            if self.fraction_train == 0.0:
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

            config["Instruction"] = "TRAIN"
            config["CKKS-context"] = public_context
            config["server-round"] = server_round

            #LIT attack
            if util.read_toml("attack-type") == "LIT":
                log(INFO, "configure_train: Computing LIT attack update...")
                lit_update = attacks.lit_attack_update(self.malicious_ids, arrays, self.num_total_clients, self.node_ids_to_partition_ids)
                config["LIT-update"] = pickle.dumps(lit_update)

            self.trained_this_round = True
        else:
            #Tell nodes from last round to send their weights
            log(INFO, "configure_train: gathering plaintext weights from previously-sampled nodes")
            self.trained_this_round = False
            config["Instruction"] = "SENDWEIGHTS"
            config["server-round"] = server_round
            self.trained_this_round = False

        record = RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: config})
        messages = self._construct_messages(record, self.current_nodes, MessageType.TRAIN)
        if util.read_toml("measure-messages"):
            self.total_message_size += sum([len(pickle.dumps(message)) for message in messages])
        return messages

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        if not valid_replies:
            return None, None
        
        if util.read_toml("measure-messages"):
            self.total_message_size += sum([len(pickle.dumps(message)) for message in replies])
        
        if self.trained_this_round and self.noise_reduction:
            #We receive ciphertexts 
            #Either aggregate or inspect
            inspecting = random.randint(0, 1)
            if inspecting:
                log(INFO, f"aggregate_train: received {len(replies)} replies; decrypting and inspecting ciphertexts")
            else:
                log(INFO, f"aggregate_train: received {len(replies)} replies; saving ciphertexts for aggregation")
            self.ids_to_ciphertexts = {}
            self.ids_to_num_examples = {}
            for reply in replies:
                id = reply.metadata.src_node_id
                records = reply.content
                if not id in self.trust_scores.keys():
                    self.trust_scores[id] = 0.5
                    self.ids_to_test_rejections[id] = 0
                if records["config"]["active"]:
                    serialized_ciphertext = records["config"]["encrypted-difference"]
                    num_examples = records["num-examples"]["num-examples"]
                    if inspecting and self.use_dp:
                        difference = np.array(ts.ckks_vector_from(self.current_ckks_context, serialized_ciphertext).decrypt())
                        self.ids_to_test_rejections[id] += int(jarque_bera(difference, 0, self.expected_std*num_examples, self.jarque_bera_alpha))
                    else:
                        #keep ciphertexts for aggregation and decryption next round
                        #it would be nice to decrypt now, but we don't know which clients 
                        #we'll include in aggregation until we inspect model weights
                        self.ids_to_ciphertexts[id] = serialized_ciphertext
                        self.ids_to_num_examples[id] = num_examples

            if inspecting and self.use_dp:
                self.current_nodes = [] #clear out list to indicate we select new clients next iteration
                
            return None, None #no global update was performed
        
        elif self.trained_this_round and not self.noise_reduction:
            log(INFO, f"aggregate_train: received {len(replies)} replies; tabulating number of examples")
            self.ids_to_num_examples = {}
            for reply in replies:
                id = reply.metadata.src_node_id
                records = reply.content
                if not id in self.trust_scores.keys():
                    self.trust_scores[id] = 0.5
                    self.ids_to_test_rejections[id] = 0
                if records["config"]["active"]:
                    num_examples = records["num-examples"]["num-examples"]
                    self.ids_to_num_examples[id] = num_examples

            return None, None #no global update was performed

                       
        else:
            #We receive plaintext model weights to inspect and aggregate
            log(INFO, f"aggregate_train: received {len(replies)} replies, inspecting and aggregating plaintext weights")
            active_clients = []
            ids_to_plaintext_weights = {}
            plaintext_weights = np.array([])
            for reply in replies:
                id = reply.metadata.src_node_id
                records = reply.content
                if not id in self.trust_scores.keys():
                    self.trust_scores[id] = 0.5
                    self.ids_to_test_rejections[id] = 0
                if records["config"]["active"]:
                    active_clients.append(id)
                    client_plaintext_weights = records["plaintext-weights"]["plaintext-weights"].numpy()
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
            aggregated_differences = [0] * plaintext_weights[0].size
            total_examples = 0
            num_examples_sq_sum = 0 #to compute expected std of aggregated_differences
            for id in active_clients:
                if self.trust_scores[id] >= 0.75 and self.ids_to_test_rejections[id] < self.max_rejections:
                    total_examples += self.ids_to_num_examples[id]
                    aggregated_weights += torch.tensor(ids_to_plaintext_weights[id]).to(device)*self.ids_to_num_examples[id]
                    if self.use_dp and self.noise_reduction:
                        aggregated_differences: ts.CKKSVector = aggregated_differences + ts.ckks_vector_from(self.current_ckks_context, self.ids_to_ciphertexts[id])
                        num_examples_sq_sum += self.ids_to_num_examples[id] ** 2

            if total_examples == 0:
                #This happens at the beginning since detection threshold is 0.75
                self.ids_to_num_examples = {}
                self.current_nodes = [] #clear out list to indicate we select new clients next iteration
                src.config.total_model_updates += 1 #increase because this round was a dp exposure despite no model update
                src.config.trust_scores = self.trust_scores
                if src.config.total_model_updates == self.max_num_updates:
                    self.fraction_train = 0
                    self.fraction_evaluate = 0
                src.config.last_update_round = server_round
                return None, None

            if self.use_dp and self.noise_reduction:  
                aggregated_differences = np.array(aggregated_differences.decrypt(), np.float32)
                if jarque_bera(aggregated_differences, 0, self.expected_std*math.sqrt(num_examples_sq_sum), self.jarque_bera_alpha):
                    log(INFO, "Components of aggregated difference vector do not follow expected distribution, aborting and starting new training round")
                    self.ids_to_num_examples = {}
                    self.current_nodes = [] #clear out list to indicate we select new clients next iteration
                    src.config.total_model_updates += 1 #increase because this round was a dp exposure despite no model update
                    src.config.trust_scores = self.trust_scores
                    if src.config.total_model_updates == self.max_num_updates:
                        self.fraction_train = 0
                        self.fraction_evaluate = 0
                    src.config.last_update_round = server_round
                    return None, None
                
                aggregated_weights = (aggregated_weights + torch.tensor(aggregated_differences).to(device)) / total_examples
            else:
                aggregated_weights /= total_examples
            
            self.ids_to_ciphertexts = {}
            self.ids_to_num_examples = {}
            aggregated_weights = ArrayRecord(util.vec_to_state_dict(model_loading.model().state_dict(), aggregated_weights.to(device)))

            # Aggregate custom metrics if aggregation fn was provided
            aggregated_metrics = self.train_metrics_aggr_fn(
                    [msg.content for msg in valid_replies],
                    self.weighted_by_key,
            )

            self.ids_to_num_examples = {}
            self.current_nodes = [] #clear out list to indicate we select new clients next iteration
            src.config.total_model_updates += 1
            src.config.trust_scores = self.trust_scores
            if src.config.total_model_updates == self.max_num_updates:
                self.fraction_train = 0
                self.fraction_evaluate = 0
            src.config.last_update_round = server_round
            gc.collect()
            return aggregated_weights, aggregated_metrics