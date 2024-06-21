#!/bin/bash

# run in host
addrs=(172.31.14.142 172.31.11.170)
image="oobleck:v12"

# Define the common part of the docker run command
# docker_command="docker run --gpus device=%d -d -v ~/code/python/:/workspace --net=host --ipc=host --device=/dev/infiniband/rdma_cm --device=/dev/infiniband/uverbs0 --ulimit memlock=-1:-1 --name oob-%d ${image} sleep infinity"

docker_command="docker run --gpus device=%d -d -v ~/code/python/:/workspace --net=host --ipc=host --ulimit memlock=-1:-1 --name oob-%d ${image} sleep infinity"

mkdir -p ./tmp/logs/install/

for addr in "${addrs[@]}"; do
    # ssh ${addr} "$(printf "${docker_command}" 0 0)
    #     $(printf "${docker_command}" 1 1)
    #     $(printf "${docker_command}" 2 2)
    #     $(printf "${docker_command}" 3 3)"

   ssh ${addr} "$(printf "${docker_command}" 0 0)" >./tmp/logs/install/sc-${addr}.log 2>&1 &
done
