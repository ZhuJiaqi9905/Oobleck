import os
from typing import Sequence
import torch
import torch.distributed as dist
import torch.multiprocessing as multiproc
import time
import argparse
import json
from oobleck.elastic.training_util import OobleckArguments
from datetime import timedelta
from oobleck.module.model import OobleckModel
from transformers.training_args import TrainingArguments as HFTrainingArguments
from oobleck.execution.dataset import OobleckDataset
import copy
from functools import reduce
import operator

class Layer:
    def __init__(
        self,
        sizes: Sequence[Sequence[int]],
        names: Sequence[str],
        send_rank: int = None,
        recv_rank: int = None,
    ) -> None:
        self._sizes = sizes
        self._names = names
        # self._parameters = []
        self._optimizer_parameters: list[torch.Tensor] = []
        self._optimizer_momentums: list[torch.Tensor] = []
        self._optimizer_variants: list[torch.Tensor] = []
        self._send_rank = send_rank
        self._recv_rank = recv_rank

    def init_tensors(self, device: str = "cuda"):
        for size in self._sizes:
            self._optimizer_parameters.append(
                torch.randn(size, dtype=torch.float32, device=device)
            )
            self._optimizer_momentums.append(
                torch.randn(size, dtype=torch.float32, device=device)
            )
            self._optimizer_variants.append(
                torch.randn(size, dtype=torch.float32, device=device)
            )

    def broadcast(self, pgs):
        src = self._ranks[0]
        pg = pgs[tuple(sorted(self._ranks))]

        # for param in self._parameters:
        #     dist.broadcast(param, src, pg)
        for op_param in self._optimizer_parameters:
            dist.broadcast(op_param, src, pg)
        for op_mom in self._optimizer_momentums:
            dist.broadcast(op_mom, src, pg)
        for op_var in self._optimizer_variants:
            dist.broadcast(op_var, src, pg)


def parse_info_file(file_path) -> dict:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            # 读取并解析JSON文件
            info = json.load(file)
            return info
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None


def test_p2p(
    global_rank: int,
    info: dict,
    send_layers: list[Layer],
    recv_layers: list[Layer],
    cpu_layers: list[Layer],
):
    start = time.time()

    requests = []
    
    # for recv
    for layer in recv_layers:
        for op_param in layer._optimizer_parameters:
            req = dist.irecv(op_param, src=layer._send_rank)
            requests.append(req)
        for op_mom in layer._optimizer_momentums:
            req = dist.irecv(op_mom, src=layer._send_rank)
            requests.append(req)
        for op_var in layer._optimizer_variants:
            req = dist.irecv(op_var, src=layer._send_rank)
            requests.append(req)
    # for send 
    for layer in send_layers:
        for op_param in layer._optimizer_parameters:
            req = dist.isend(op_param, dst=layer._recv_rank)
            requests.append(req)
        for op_mom in layer._optimizer_momentums:
            req = dist.isend(op_mom, dst=layer._recv_rank)
            requests.append(req)
        for op_var in layer._optimizer_variants:
            req = dist.isend(op_var, dst=layer._recv_rank)
            requests.append(req)    
    # for cpu copy
    local_layers = []
    for layer in cpu_layers:
        for op_param in layer._optimizer_parameters:
            t = op_param.to('cuda')
            local_layers.append(t)
        for op_mom in layer._optimizer_momentums:
            t = op_mom.to('cuda')
            local_layers.append(t)
        for op_var in layer._optimizer_variants:
            t = op_var.to('cuda')
            local_layers.append(t)

    for req in requests:
        req.wait()

    dist.barrier()
    torch.cuda.synchronize()
    end = time.time()
    return (end - start) * 1000


def run(
    local_rank: int,
    global_rank: int,
    warmup_times: int,
    repeat_times: int,
    info: dict,
    transformer_layer: Layer,
):
    torch.cuda.set_device(local_rank)
    print(f"local_rank: {local_rank}")
    # init tensors
    send_info = info["send_info"]
    recv_info = info["recv_info"]
    send_chunk_sizes = info["send_chunk_sizes"]
    recv_chunk_sizes = info["recv_chunk_sizes"]
    send_layers: list[Layer] = []

    total_send_size = 0
    total_recv_size = 0
    for i in range(len(send_info[global_rank])):
        if i == global_rank:
            continue
        sizes = copy.deepcopy(transformer_layer._sizes)
        print(f"sizes: {sizes}")

        for s in sizes:
            s = list(s)
            s[-1] = s[-1] // send_chunk_sizes[global_rank][i]
            total_send_size += reduce(operator.mul, s, 1)
        for _ in range(send_info[global_rank][i]):
            send_layers.append(Layer(sizes, transformer_layer._names, global_rank, i))
            send_layers[-1].init_tensors()

    recv_layers: list[Layer] = []
    for i in range(len(recv_info[global_rank])):
        if i == global_rank:
            continue
        sizes = copy.deepcopy(transformer_layer._sizes)
        for s in sizes:
            s = list(s)
            s[-1] = s[-1] // recv_chunk_sizes[global_rank][i]
            total_recv_size += reduce(operator.mul, s, 1) 
        for _ in range(recv_info[global_rank][i]):
            recv_layers.append(Layer(sizes, transformer_layer._names, i, global_rank))
            recv_layers[-1].init_tensors()

    cpu_layers: list[Layer] = []
    if send_info[global_rank][global_rank] != 0:
        for _ in range(send_info[global_rank][global_rank]):
            cpu_layers.append(Layer(transformer_layer._sizes, transformer_layer._names))
            cpu_layers[-1].init_tensors(device="cpu")

    print(f"send layers")
    for layer in send_layers:
        for i in range(len(layer._sizes)):
            print(f"{layer._names[i]}: {layer._sizes[i]}")
    print(f"recv layers")
    for layer in recv_layers:
        for i in range(len(layer._sizes)):
            print(f"{layer._names[i]}: {layer._sizes[i]}")    

    print("init tensor success")

    for _ in range(warmup_times):
        _ = test_p2p(
            global_rank, info, send_layers, recv_layers, cpu_layers
        )
    total_time = 0
    for _ in range(repeat_times):
        total_time += test_p2p(
            global_rank, info, send_layers, recv_layers, cpu_layers
        )
    avg_time_result_in_ms = total_time / repeat_times
    total_send_size *= 12
    total_recv_size *= 12
    if global_rank == 0:
        print(
            f"(GPU, Rank {global_rank} | Time(averaged {repeat_times} times) = {avg_time_result_in_ms:.2f} ms | send_size = {total_send_size} B | recv_size = {total_recv_size} B"
        )


def init_process(
    local_rank: int,
    global_rank: int,
    world_size: int,
    master_ip: str,
    master_port: str,
    warmup_times: int,
    repeat_times: int,
    info: dict,
    model: str,
    fn,
):
    print(f"global_rank: {global_rank}. world_size: {world_size}")
    if global_rank >= world_size:
        return

    transformer_layer = get_model_transformer_layer(model)
    init_method = "tcp://"
    init_method += master_ip + ":" + master_port

    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=global_rank,
        init_method=init_method,
        timeout=timedelta(minutes=5),
    )
    print("init process group success")
    fn(local_rank, global_rank, warmup_times, repeat_times, info, transformer_layer)


def get_model_transformer_layer(model: str) -> Layer:
    if model == "gpt3_2_7B":
        config_path = "/workspace/Oobleck/examples/gpt3_2_7B.yaml"
    elif model == "gpt3_1_3B":
        config_path = "/workspace/Oobleck/examples/gpt3_1_3B.yaml"
    elif model == "gpt3_6_7B":
        config_path = "/workspace/Oobleck/examples/gpt3_6_7B.yaml"
    elif model == "gpt3_350M":
        config_path = "/workspace/Oobleck/examples/gpt3_350M.yaml"
    elif model == "gpt3_13B":
        config_path = "/workspace/Oobleck/examples/gpt3_13B.yaml"
    elif model == "bert_340M":
        config_path = "/workspace/Oobleck/examples/bert_340M.yaml"
    else:
        raise Exception(f"No config path for model: {model}")
    args = OobleckArguments.load_yaml(config_path)
    training_args = {
        "output_dir": f"/workspace/Oobleck/tmp/output/{args.model.model_name}-{args.model.model_tag}",
        "per_device_train_batch_size": args.job.microbatch_size,
        "no_cuda": True,  # don't use cuda in HFTrainingArguments
        "log_level": "error",  # omit warning messages from HFTrainingArguments
        # do not set gradient_accumulation_steps in HFTrainingArguments
        "max_steps": args.job.steps,
    }
    hf_training_args: HFTrainingArguments = HFTrainingArguments(**training_args)
    dataset = OobleckDataset(
        args.model.model_name,
        args.model.dataset_path,
        args.model.dataset_name,
        (
            args.model.model_args["n_positions"]
            if "n_positions" in args.model.model_args
            else None
        ),
    )
    model = OobleckModel(
        args.model.model_name,
        dataset.sample,
        hf_training_args,
        args.model.model_tag,
        args.model.model_args,
    )
    sizes = []
    names = []
    for name, param in model.layers[1].named_parameters():
        sizes.append(param.size())
        names.append(name)
    return Layer(sizes, names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--info-file", type=str, help="The path to the JSON file")
    parser.add_argument("--gpus-per-node", type=int)
    parser.add_argument("--node-rank", type=int)
    parser.add_argument("--master-ip", type=str)
    parser.add_argument("--master-port", type=str)
    parser.add_argument("--warmup-times", type=int)
    parser.add_argument("--repeat-times", type=int)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    print(args.info_file)
    info = parse_info_file(args.info_file)
    if not info:
        print("parse json error")
        exit()
    print("parse info-file success")
    world_size: int = len(info["send_info"])
    gpus_per_node = args.gpus_per_node
    node_rank = args.node_rank

    multiproc.set_start_method("spawn")

    processes = []
    print(f"gpus_per_node: {gpus_per_node}")
    for local_rank in range(gpus_per_node):
        global_rank = gpus_per_node * node_rank + local_rank
        print(f"local_rank: {local_rank}. global_rank: {global_rank}")
        p = multiproc.Process(
            target=init_process,
            args=(
                local_rank,
                global_rank,
                world_size,
                args.master_ip,
                args.master_port,
                args.warmup_times,
                args.repeat_times,
                info,
                args.model,
                run,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
