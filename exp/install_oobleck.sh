#!/bin/bash
# run in container
# addrs=("172.31.11.113" "172.31.9.213")
# ports=(2220)

addrs=("172.21.0.42" "172.21.0.46" "172.21.0.90" "172.21.0.92")
ports=(2220 2221 2222 2223)

mkdir -p ./tmp/logs/install/

for port in "${ports[@]}"; do
    for addr in "${addrs[@]}"; do
        echo "in ${addr}:${port}"
        ssh -p ${port} ${addr} -o "StrictHostKeyChecking no" "cd /workspace/Oobleck && ./exp/local_install_oobleck.sh" > ./tmp/logs/install/${addr}-${port}.log 2>&1 &
    done
done