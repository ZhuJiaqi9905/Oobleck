import os
from typing import Sequence
import torch
import torch.distributed as dist
import torch.multiprocessing as multiproc
import time
import argparse
import json

from datetime import timedelta


class Layer:
    def __init__(
        self, sizes: Sequence[Sequence[int]], ranks: Sequence[int], names: Sequence[str]
    ) -> None:
        self._sizes = sizes
        self._ranks = ranks
        self._names = names
        self._gradients = []
        self._parameters = []
        self._optimizer_gradients = []
        self._optimizer_momentums = []
        self._optimizer_variants = []

    # def init_process_group(self):
    #     self._pg = dist.new_group(self._ranks)

    def init_tensors(self, global_rank: int):
        if global_rank not in self._ranks:
            return
        for size in self._sizes:
            self._parameters.append(torch.randn(size, dtype=torch.float16).cuda())
            self._gradients.append(torch.randn(size, dtype=torch.float16).cuda())
            self._optimizer_gradients.append(
                torch.randn(size, dtype=torch.float32).cuda()
            )
            self._optimizer_momentums.append(
                torch.randn(size, dtype=torch.float32).cuda()
            )
            self._optimizer_variants.append(
                torch.randn(size, dtype=torch.float32).cuda()
            )

    def broadcast(self, pgs):
        src = self._ranks[0]
        pg = pgs[tuple(sorted(self._ranks))]
        
        for param in self._parameters:
            dist.broadcast(param, src, pg)
        for grad in self._gradients:
            dist.broadcast(grad, src, pg)
        for op_grad in self._optimizer_gradients:
            dist.broadcast(op_grad, src, pg)
        for op_mom in self._optimizer_momentums:
            dist.broadcast(op_mom, src, pg)
        for op_var in self._optimizer_variants:
            dist.broadcast(op_var, src, pg)


def parse_layers(file_path) -> list[Layer] | None:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            # 读取并解析JSON文件
            data = json.load(file)
            data_layers = data["layers"]
            layers = []
            for l in data_layers:
                layers.append(Layer(l["sizes"], l["ranks"], l["names"]))
            return layers
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def test_broadcast(global_rank: int, layers: Sequence[Layer], pgs):
    start = time.time()
    for id, layer in enumerate(layers):
        if global_rank in layer._ranks:
            layer.broadcast(pgs)
    dist.barrier()
    torch.cuda.synchronize()
    end = time.time()
    return (end - start) * 1000


def run(local_rank, global_rank, layers: Sequence[Layer]):
    warmup_times = 0
    repeat_times = 1
    torch.cuda.set_device(local_rank)
    # init tensors
    for layer in layers:
        layer.init_tensors(global_rank)
    print("init tensor success")

    # init pgs
    pgs = {}
    unique_ranks = {tuple(sorted(layer._ranks)) for layer in layers}
    for ranks in unique_ranks:
        pgs[ranks] = dist.new_group(list(ranks))
        print(f"new pg: {ranks}") 
    print("init pgs success")

    for _ in range(warmup_times):
        _ = test_broadcast(global_rank, layers, pgs)
    total_time = 0
    for _ in range(repeat_times):
        total_time += test_broadcast(global_rank, layers, pgs)
    avg_time_result_in_ms = total_time / repeat_times
    if global_rank == 0:
        print(
            f"(GPU, Rank {global_rank} | Time(averaged {repeat_times} times) = {avg_time_result_in_ms:.2f} ms"
        )


def init_process(
    local_rank: int,
    global_rank: int,
    world_size: int,
    master_ip: str,
    master_port: str,
    layers: Sequence[Layer],
    fn,
):
    init_method = "tcp://"
    init_method += master_ip + ":" + master_port

    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=global_rank,
        init_method=init_method,
        timeout=timedelta(minutes=3),
    )
    print("init process group success")
    fn(local_rank, global_rank, layers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layer-file", type=str, help="The path to the JSON file of layer"
    )
    parser.add_argument("--gpus-per-node", type=int)
    parser.add_argument("--num-nodes", type=int)
    parser.add_argument("--node-rank", type=int)
    parser.add_argument("--master-ip", type=str)
    parser.add_argument("--master-port", type=str)

    args = parser.parse_args()
    print(args.layer_file)
    layers = parse_layers(args.layer_file)
    if not layers:
        print("parse json error")
        exit()
    print("parse layers success")
    gpus_per_node = args.gpus_per_node
    num_nodes = args.num_nodes
    node_rank = args.node_rank

    multiproc.set_start_method("spawn")
    world_size: int = gpus_per_node * num_nodes
    processes = []
    for local_rank in range(gpus_per_node):
        global_rank = gpus_per_node * node_rank + local_rank
        p = multiproc.Process(
            target=init_process,
            args=(
                local_rank,
                global_rank,
                world_size,
                args.master_ip,
                args.master_port,
                layers,
                run,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
