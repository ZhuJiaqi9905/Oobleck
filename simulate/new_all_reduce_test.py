import os
from typing import Sequence
import torch
import torch.distributed as dist
import torch.multiprocessing as multiproc
import time
import argparse
import json
from bitmap import BitMap
from datetime import timedelta


class DistInfo:
    def __init__(self, info) -> None:
        self._model: str = info["model"]
        self._microbatch_size: int = info["microbatch_size"]
        self._world_size: int = info["world_size"]
        self._ranks_info: list[dict] = []
        self._layers_info: list[dict] = info["layers_info"]

        for layers in self._layers_info:
            layers["ranks"] = []

        for _ in range(self._world_size):
            self._ranks_info.append({"pipeline_id": 0, "layers": []})
        for pipeline_id in range(len(info["pipelines"])):
            pipeline: dict[str, list[int]] = info["pipelines"][pipeline_id]
            for layer_id, ranks in pipeline["layers"].items():
                for rank in ranks:
                    self._ranks_info[rank]["pipeline_id"] = pipeline_id
                    self._ranks_info[rank]["layers"].append(int(layer_id))
                    if rank not in self._layers_info[layer_id]["ranks"]:
                        self._layers_info[layer_id]["ranks"].append(rank)
        for layer in self._layers_info:
            layer["ranks"].sort()
        print(f"ranks_info: {self._ranks_info}")
        print(f"layers_info: {self._layers_info}")


class Engine:
    def __init__(self, rank: int, info: DistInfo) -> None:
        self._rank = rank  # global rank
        self._layers: dict[int, Layer] = {}
        self._pgs: dict[tuple[int], dist.ProcessGroup | None] = {}
        self._dist_info = info
        for layer_id in info._ranks_info[rank]["layers"]:
            self._layers[layer_id] = Layer(info._layers_info[layer_id]["sizes"])
            ranks: tuple[int] = tuple(info._layers_info[layer_id]["ranks"])
            if ranks not in self._pgs:
                self._pgs[ranks] = None

        for ranks in self._pgs.keys():
            self._pgs[rank] = dist.new_group(list(ranks))
            print(f"new pg: {ranks}")

        for layer_id, layer in self._layers.items():
            layer._pg = self._pgs[tuple(info._layers_info[layer_id]["ranks"])]


class Layer:
    def __init__(self, sizes: Sequence[Sequence[int]], names: Sequence[str]) -> None:
        self._sizes = sizes
        self._names = names
        self._gradients = []
        self._pg = None
        for size in self._sizes:
            self._gradients.append(torch.randn(size, dtype=torch.float16).cuda())

    def allreduce(self):
        for grad in self._gradients:
            dist.all_reduce(grad, group=self._pg)



def parse_json(file_path) -> DistInfo | None:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            # 读取并解析JSON文件
            data = json.load(file)
            return DistInfo(data)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def test_allreduce(engine: Engine):
    start = time.time()

    for layer in reversed(engine._layers.values()):
        layer.allreduce()

    dist.barrier()
    torch.cuda.synchronize()
    end = time.time()
    duration = (end - start) * 1000
    print(f"duration: {duration} ms")
    return duration


def run(local_rank: int, global_rank: int, info: DistInfo):
    warmup_times = 0
    repeat_times = 1
    torch.cuda.set_device(local_rank)

    engine = Engine(global_rank, info)


    for _ in range(warmup_times):
        _ = test_allreduce(engine)
    total_time = 0
    for _ in range(repeat_times):
        total_time += test_allreduce(engine)
    
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
    info: DistInfo,
    fn,
):
    if global_rank >= world_size:
        return

    init_method = "tcp://"
    init_method += master_ip + ":" + master_port

    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=global_rank,
        init_method=init_method,
        timeout=timedelta(minutes=3),
    )
    print("init distributed process group success")
    fn(local_rank, global_rank, info)


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
    info = parse_json(args.layer_file)
    if not info:
        print("parse json error")
        exit()
    print("parse json success")

    gpus_per_node = args.gpus_per_node
    num_nodes = args.num_nodes
    node_rank = args.node_rank

    multiproc.set_start_method("spawn")
    world_size: int = info._world_size
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
                info,
                run,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
