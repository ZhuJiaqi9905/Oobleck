#!/bin/bash
# run in container. only need to install on oob-0
# addrs=(172.21.0.42 172.21.0.46 172.21.0.47 172.21.0.90 172.21.0.91 172.21.0.92)
# ports=(2220 2221 2222 2223)

# aws
addrs=(172.31.44.82 172.31.35.195 172.31.40.219 172.31.45.26)
ports=(2220 2221 2222 2223 2224 2225 2226 2227)

# addrs=(172.31.11.113 172.31.9.213)
# ports=(2220)

for port in "${ports[@]}"; do
    for addr in "${addrs[@]}"; do
        echo "in ${addr}:${port}"
        ssh -p ${port} ${addr} -o "StrictHostKeyChecking no" "cd /workspace/Oobleck && ./exp/local_kill.sh"
    done
done