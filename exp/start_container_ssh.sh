#!/bin/bash

# run in host
addrs=(172.21.0.42 172.21.0.46 172.21.0.90 172.21.0.92)
ssh_ports=(2220 2221 2222 2223)

num_containers=${#ssh_ports[@]}

for addr in "${addrs[@]}"; do
    for((i=0;i<${num_containers};i++)); do
        echo "${addr}: ${ssh_ports[${i}]}"
        ssh ${addr} "docker exec oob-${i} bash -c 'sed -i \"s/^Port 2220/Port ${ssh_ports[${i}]}/\" /etc/ssh/sshd_config && service ssh start'"
    done
done

