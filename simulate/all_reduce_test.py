import os
import torch
import torch.distributed as dist
import torch.multiprocessing as multiproc
import time
import argparse
from datetime import timedelta


def run(local_rank, global_rank):
    warmup_times = 20
    repeat_times = 100
    torch.cuda.set_device(local_rank)

    for i in range(1, 12):
        data_size_in_mb = 2**i
        data_size = data_size_in_mb * 1024 * 1024 // 2

        tensor = torch.ones(data_size, dtype=torch.float16).cuda()

        ## warmup
        for i in range(warmup_times):
            dist.all_reduce(tensor=tensor)

        dist.barrier()
        torch.cuda.synchronize()
        
        start = time.time()
        for i in range(repeat_times):
            dist.all_reduce(tensor=tensor)
        dist.barrier()
        torch.cuda.synchronize()
        end = time.time()

        avg_time_result_in_ms = (end - start) * 1000 / repeat_times
        bandwidth_in_gb_per_second = (data_size_in_mb / 1024) / (
            avg_time_result_in_ms / 1000
        )
        if global_rank == 0:
            print(
                f"(GPU, Rank {global_rank} | Time(averaged {repeat_times} times) = {avg_time_result_in_ms:.2f} ms, data_size = {data_size_in_mb:.2f} MB, bandwidth = {bandwidth_in_gb_per_second:.2f} GB/s"
            )


def init_process(
    local_rank, global_rank, world_size, master_ip: str, master_port: str, fn
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
    fn(local_rank, global_rank)


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

    gpus_per_node = args.gpus_per_node
    num_nodes = args.num_nodes
    node_rank = args.node_rank
    multiproc.set_start_method("spawn")
    world_size = gpus_per_node * num_nodes

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
                run,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
