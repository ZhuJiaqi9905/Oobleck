#!/bin/bash

# run in host
# addrs=(172.31.14.142 172.31.11.170)
addrs=(172.31.10.88 172.31.8.235)

image="oobleck:v14"

# Define the common part of the docker run command
# docker_command="docker run --gpus device=%d -d -v ~/code/python/:/workspace --net=host --ipc=host --device=/dev/infiniband/rdma_cm --device=/dev/infiniband/uverbs0 --ulimit memlock=-1:-1 --name oob-%d ${image} sleep infinity"

docker_command="docker run --gpus device=%d -d -v ~/code/python/:/workspace --net=host --ipc=host --ulimit memlock=-1:-1 --name oob-%d ${image} sleep infinity"

mkdir -p ./tmp/logs/install/

for addr in "${addrs[@]}"; do
    # ssh ${addr} "$(printf "${docker_command}" 0 0)
    #     $(printf "${docker_command}" 1 1)
    #     $(printf "${docker_command}" 2 2)
    #     $(printf "${docker_command}" 3 3)" >./tmp/logs/install/sc-${addr}.log 2>&1 &


    ssh ${addr} "$(printf "${docker_command}" 0 0)
        $(printf "${docker_command}" 1 1)
        $(printf "${docker_command}" 2 2)
        $(printf "${docker_command}" 3 3)
        $(printf "${docker_command}" 4 4)
        $(printf "${docker_command}" 5 5)
        $(printf "${docker_command}" 6 6)
        $(printf "${docker_command}" 7 7)" >./tmp/logs/install/sc-${addr}.log 2>&1 &     



#    ssh ${addr} "$(printf "${docker_command}" 0 0)" >./tmp/logs/install/sc-${addr}.log 2>&1 &
done
