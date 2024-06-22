#!/bin/bash
# run in container. only need to install on oob-0
# addrs=("172.31.11.113" "172.31.9.213")
# ports=(2220)

addrs=(172.31.10.88 172.31.8.235)
ports=(2220 2221 2222 2223 2224 2225 2226 2227)


for port in "${ports[@]}"; do
    for addr in "${addrs[@]}"; do
        echo "in ${addr}:${port}"
        ssh -p ${port} ${addr} -o "StrictHostKeyChecking no" "cd /workspace/Oobleck && ./exp/local_kill.sh"
    done
done