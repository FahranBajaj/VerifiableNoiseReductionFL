#!/bin/bash

clipping_norms=(1 2 0.5 4 0.25)
for norm in ${clipping_norms[@]}
do
    echo "Bash script process id: $$"
    flwr run . --stream --federation-config "num-supernodes=45 client-resources-num-cpus=1" --run-config "max-norm=${norm}"
done