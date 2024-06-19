#!/bin/bash

addrs=(172.21.0.42)
image="oobleck:v11"

# Define the common part of the docker run command
docker_command="docker run --gpus device=%d -d -v ~/code/python/:/workspace --net=host --ipc=host --device=/dev/infiniband/rdma_cm --device=/dev/infiniband/uverbs0 --ulimit memlock=-1:-1 --name oob-%d ${image} sleep infinity"

for addr in "${addrs[@]}"; do
    ssh ${addr} "$(printf "${docker_command}" 0 0)
        $(printf "${docker_command}" 1 1)
        $(printf "${docker_command}" 2 2)
        $(printf "${docker_command}" 3 3)"
done
