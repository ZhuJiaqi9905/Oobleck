#!/bin/bash

# run in host
addrs=(172.21.0.42 172.21.0.46 172.21.0.90 172.21.0.92)
ports=(2220 2221 2222 2223)

# addrs=(172.31.10.88 172.31.8.235)
# ports=(2220 2221 2222 2223 2224 2225 2226 2227)


# addrs=(172.31.11.113 172.31.9.213)
# ports=(2220)

num_containers=${#ports[@]}

for addr in "${addrs[@]}"; do
    for((i=0;i<${num_containers};i++)); do
        echo "${addr}: ${ports[${i}]}"
        ssh ${addr} -o "StrictHostKeyChecking no" "docker exec oob-${i} bash -c 'sed -i \"s/^Port 2220/Port ${ports[${i}]}/\" /etc/ssh/sshd_config && service ssh start'"
    done
done

