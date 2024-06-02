#!/bin/bash

world_size=10
if [ "$world_size" -eq 8 ]; then
    nodes="10.20.23.42 10.20.23.42 10.20.23.46 10.20.23.46 10.20.23.90 10.20.23.90 10.20.23.92 10.20.23.92"
    node_ports="2220 2221 2220 2221 2220 2221 2220 2221"
elif [ "$world_size" -eq 9 ]; then
    nodes="10.20.23.42 10.20.23.42 10.20.23.42 10.20.23.46 10.20.23.46 10.20.23.90 10.20.23.90 10.20.23.92 10.20.23.92"
    node_ports="2220 2221 2222 2220 2221 2220 2221 2220 2221"
elif [ "$world_size" -eq 10 ]; then
    nodes="10.20.23.42 10.20.23.42 10.20.23.42 10.20.23.46 10.20.23.46 10.20.23.46 10.20.23.90 10.20.23.90 10.20.23.92 10.20.23.92"
    node_ports="2220 2221 2222 2220 2221 2222 2220 2221 2220 2221"
elif [ ${world_size} -eq 11 ]; then
    nodes="10.20.23.42 10.20.23.42 10.20.23.42 10.20.23.46 10.20.23.46 10.20.23.46 10.20.23.90 10.20.23.90 10.20.23.90 10.20.23.92 10.20.23.92"
    node_ports="2220 2221 2222 2220 2221 2222 2220 2221 2222 2220 2221"
elif [ ${world_size} -eq 12 ]; then
    nodes="10.20.23.42 10.20.23.42 10.20.23.42 10.20.23.46 10.20.23.46 10.20.23.46 10.20.23.90 10.20.23.90 10.20.23.90 10.20.23.92 10.20.23.92 10.20.23.92"
    node_ports="2220 2221 2222 2220 2221 2222 2220 2221 2222 2220 2221 2222"
elif [ "$world_size" -eq 13 ]; then 
    nodes="10.20.23.42 10.20.23.42 10.20.23.42 10.20.23.42 10.20.23.46 10.20.23.46 10.20.23.46 10.20.23.90 10.20.23.90 10.20.23.90 10.20.23.92 10.20.23.92 10.20.23.92"
    node_ports="2220 2221 2222 2223 2220 2221 2222 2220 2221 2222 2220 2221 2222"
elif [ ${world_size} -eq 14 ]; then
    nodes="10.20.23.42 10.20.23.42 10.20.23.42 10.20.23.42 10.20.23.46 10.20.23.46 10.20.23.46 10.20.23.46 10.20.23.90 10.20.23.90 10.20.23.90 10.20.23.92 10.20.23.92 10.20.23.92"
    node_ports="2220 2221 2222 2223 2220 2221 2222 2223 2220 2221 2222 2220 2221 2222"
elif [ ${world_size} -eq 15 ]; then
    nodes="10.20.23.42 10.20.23.42 10.20.23.42 10.20.23.42 10.20.23.46 10.20.23.46 10.20.23.46 10.20.23.46 10.20.23.90 10.20.23.90 10.20.23.90 10.20.23.90 10.20.23.92 10.20.23.92 10.20.23.92"
    node_ports="2220 2221 2222 2223 2220 2221 2222 2223 2220 2221 2222 2223 2220 2221 2222"
elif [ ${world_size} -eq 16 ]; then
    nodes="10.20.23.42 10.20.23.42 10.20.23.42 10.20.23.42 10.20.23.46 10.20.23.46 10.20.23.46 10.20.23.46 10.20.23.90 10.20.23.90 10.20.23.90 10.20.23.90 10.20.23.92 10.20.23.92 10.20.23.92 10.20.23.92"
    node_ports="2220 2221 2222 2223 2220 2221 2222 2223 2220 2221 2222 2223 2220 2221 2222 2223"
fi



python -m oobleck.run \
--config_path ./examples/gpt3_350M.yaml \
--node_ips ${nodes} \
--node_ports ${node_ports} \
--master_ip 10.20.23.42 \
--master_port 60000
