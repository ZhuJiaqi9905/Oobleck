#!/bin/bash
# run in container
addrs=("172.21.0.42" "172.21.0.46")
# ports=(2220 2221 2222 2223)
ports=(2220)

for port in "${ports[@]}"; do
    for addr in "${addrs[@]}"; do
        echo "in ${addr}:${port}"
        ssh -p ${port} ${addr} -o "StrictHostKeyChecking no" "cd /workspace/Oobleck && ./exp/local_install_oobleck.sh" 
    done
done