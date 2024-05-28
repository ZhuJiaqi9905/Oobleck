#!/bin/bash
docker run --gpus device=0 -d  -v ~/code/python/:/workspace --net=host --ipc=host --device=/dev/infiniband/rdma_cm --device=/dev/infiniband/uverbs0 --ulimit memlock=-1:-1  --name oob-0 oobleck:v10  sleep infinity
docker run --gpus device=1 -d  -v ~/code/python/:/workspace --net=host --ipc=host --device=/dev/infiniband/rdma_cm --device=/dev/infiniband/uverbs0 --ulimit memlock=-1:-1  --name oob-1 oobleck:v10  sleep infinity
docker run --gpus device=2 -d  -v ~/code/python/:/workspace --net=host --ipc=host --device=/dev/infiniband/rdma_cm --device=/dev/infiniband/uverbs0 --ulimit memlock=-1:-1  --name oob-2 oobleck:v10  sleep infinity
docker run --gpus device=3 -d  -v ~/code/python/:/workspace --net=host --ipc=host --device=/dev/infiniband/rdma_cm --device=/dev/infiniband/uverbs0 --ulimit memlock=-1:-1  --name oob-3 oobleck:v10  sleep infinity

