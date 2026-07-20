#!/bin/bash

learning_rates=(0.001 0.003 0.01 0.03 0.1)
for learning_rate in ${learning_rates[@]}
do
    flwr run . --stream --federation-config "num-supernodes=100 client-resources-num-cpus=1" --run-config "learning-rate=${learning_rate}"
done

local_epochs=(1 2 4 8 16)
for epochs in ${local_epochs[@]}
do
    flwr run . --stream --federation-config "num-supernodes=100 client-resources-num-cpus=1" --run-config "local-epochs=${epochs}"
done

clipping_norms=(0.25 0.5 1 2 4)
for norm in ${clipping_norms[@]}
do
    flwr run . --stream --federation-config "num-supernodes=100 client-resources-num-cpus=1" --run-config "max-norm=${norm}"
done