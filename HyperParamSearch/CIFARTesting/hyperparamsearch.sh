#!/bin/bash

clipping_norms=(0.25 0.5 1 2 4)
for norm in ${clipping_norms[@]}
do
    flwr run . --stream --federation-config "num-supernodes=100 client-resources-num-cpus=1" --run-config "max-norm=${norm}"
done