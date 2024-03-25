#!/bin/bash
docker run --gpus all -d  -v ~/code/python/:/workspace --net=host --ipc=host --device=/dev/infiniband/rdma_cm --device=/dev/infiniband/uverbs0 --name oob oobleck:v7  sleep infinity
