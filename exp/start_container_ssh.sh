#!/bin/bash

# run in host
addrs=(172.31.14.142 172.31.11.170)
ssh_ports=(2220)
num_containers=${#ssh_ports[@]}

for addr in "${addrs[@]}"; do
    for((i=0;i<${num_containers};i++)); do
        ssh ${addr} "docker exec oob-${i} bash -c 'sed -i \"s/^Port 2220/Port ${ssh_ports[${i}]}/\" /etc/ssh/sshd_config && service ssh start'"
    done
done

