import gc
import json
import math
import time
from pathlib import Path

import os
import torch
import torch.distributed as dist
import torch.fx
from deepspeed.utils.logging import LoggerFactory

from oobleck.elastic.training_util import OobleckArguments
from oobleck.execution.dataset import OobleckDataset
from oobleck.execution.layer import init_tensors
from oobleck.module.model import OobleckModel

PROFILE_CACHE = "/workspace/Oobleck/tmp/profiles"
num_warmup = 1
num_iteration = 3

logger = LoggerFactory.create_logger("oobleck_profiler")


class Profiler:
    """Oobleck Profiler that profiles execution latency, allreduce latency in node,
    allreduce latency across node for each layer of the model.

    To support large model profiling, we offload parameters layer by layer.
    """

    def __init__(
        self,
        model: OobleckModel,
        num_workers_per_node: int,
        world_size: int,
    ):
        self.model = model
        self.num_workers_per_node = num_workers_per_node
        self.world_size = world_size
    
    def profile_execution_layers(self, batch_size: int) -> list[dict[str, float]]:
        '''
        batch_size is micro_batch_size
        '''
        assert dist.is_initialized()
        import copy

        results: list[list[int]] = [
            [0.0, 0.0, 0.0, 0.0] for _ in range(len(self.model.layers))
        ]
        if dist.get_rank() == 0:
            for i in range(num_warmup + 1):
                logger.info(f"Profiling layer execution latency: {i} iteration")
                input = tuple(
                    [
                        t.detach().clone().to("cuda")
                        for t in self.model.sample_inputs.values()
                    ]
                )

                # Implement a batch
                if batch_size > 1:
                    new_input = []
                    for i in range(len(input)):
                        repeat = [batch_size] + [1] * (len(input[i].shape) - 1)
                        new_input.append(input[i].repeat(repeat))
                    input = tuple(new_input)

                for idx, layer in enumerate(self.model.layers):
                    start_mem = torch.cuda.memory_allocated()

                    gpu_layer = copy.deepcopy(layer).to("cuda")
                    torch.cuda.synchronize()
                    end_mem = torch.cuda.memory_allocated()
                    # the memory of layer
                    model_mem = end_mem - start_mem
                    # forward time 
                    start = time.time_ns()
                    with torch.no_grad():
                        output = gpu_layer(*input)
                        torch.cuda.synchronize()
                    end = time.time_ns()

                    end_mem2 = torch.cuda.memory_allocated()
                    activation_mem = end_mem2 - end_mem

                    if isinstance(output, tuple):
                        input = tuple(
                            [
                                t.detach().clone() if isinstance(t, torch.Tensor) else t
                                for t in output
                            ]
                        )
                    elif isinstance(output, torch.Tensor):
                        input = output.detach().clone()

                    forward = (end - start) / 1_000_000

                    output = None
                    gpu_layer = None
                    gc.collect()
                    torch.cuda.empty_cache()

                    if i < num_warmup:
                        continue

                    results[idx][0] = forward
                    results[idx][1] = forward * 3
                    results[idx][2] = model_mem * 2
                    results[idx][3] = activation_mem * 2
        logger.info("before barrier")
        dist.barrier()
        logger.info("after barrier")
        # broadcast results to other nodes
        # 2d tensor, for each layer, multiple allreduce with different number of nodes
        results: torch.Tensor = torch.tensor(
            results, dtype=torch.float32, device="cuda", requires_grad=False
        )
        logger.info("before broadcast")
        dist.broadcast(results, 0)
        logger.info("after broadcast")
        return [
            {
                "forward": result[0],
                "backward": result[1],
                "mem_required": (result[2], result[3]),
            }
            for result in results.tolist()
        ]

    @staticmethod
    def profile_allreduce_layer(
        layer: torch.fx.GraphModule, process_group: dist.ProcessGroup
    ) -> float:
        numel = sum([p.numel() for p in layer.parameters()])
        tensor = torch.zeros(numel, dtype=torch.float16, device="cuda")

        dist.barrier(process_group)
        start = time.time_ns()
        dist.all_reduce(tensor, group=process_group)
        dist.barrier(process_group)
        end = time.time_ns()

        del tensor
        return (end - start) / 1_000_000

    def profile_allreduce_across_nodes(self) -> list[dict[int, float]]:
        """Profile allreduce latency across nodes,
        \# nodes = 2, 3, ... N.
        Actual measurement is done only on global rank 0,
        later others will receive the result from the rank.

        Returns:
            list[dict[int, float]]: A list of allreduce latency,
            where key is the number of nodes and value is the latency,
            for every layer.
        """
        assert dist.is_initialized()
        logger.info(f"Profile allreduce across {self.world_size} nodes latency")

        ranks = list(range(0, dist.get_world_size()))

        process_groups: list[tuple(bool, dist.ProcessGroup)] = []
        # each node use one gpu to allreduce. for example 2 node, 4 GPU per node: process groups are [[0, 4], [1, 5], [2, 6], [3, 7]]
        for i in range(0, len(ranks), self.num_workers_per_node):
            pg_ranks = ranks[i : i + self.num_workers_per_node]
            process_groups.append(
                (dist.get_rank() in pg_ranks, dist.new_group(pg_ranks))
            )

        results: list[list[int]] = [
            [0] * len(process_groups) for _ in range(len(self.model.layers))
        ]
        for layer_index, layer in enumerate(self.model.layers):
            for pg_index, (should_run, pg) in enumerate(process_groups):
                if should_run:
                    results[layer_index][pg_index] = Profiler.profile_allreduce_layer(
                        layer, pg
                    )

        dist.barrier()
        # broadcast results to other nodes
        # 2d tensor, for each layer, multiple allreduce with different number of nodes
        results: torch.Tensor = torch.tensor(
            results, dtype=torch.float32, device="cuda", requires_grad=False
        )
        dist.broadcast(results, 0)

        return [
            {len(ranks[:i]) + 1: result[i] for i in range(len(result))}
            for result in results.tolist()
        ]

    def profile_allreduce_in_node(
        self, num_gpus_per_node: int
    ) -> list[dict[int, float]]:
        """Profile allreduce latency between GPUs in node,
        \# nodes = 1, 2, 4, ....
        Actual measurement is done only on global rank 0,
        later others will receive the result from the rank.

        Returns:
            list[dict[int, float]]: A list of allreduce latency,
            where key is the number of GPUs and value is the latency,
            for every layer.
        """
        assert dist.is_initialized()
        logger.info(f"Profile allreduce within a node latency")
        # assume 4 gpus per node, then ranks is [0, 1, 2, 3], num_gpus_list is [1, 2, 4] 
        num_gpus_list = [2**i for i in range(int(math.log2(num_gpus_per_node)) + 1)]
        ranks = list(range(num_gpus_per_node))
        
        # process_groups are [(bool, [0]), (bool, [0, 1]), (bool, [0, 1, 2, 3])]
        process_groups: list[tuple(bool, dist.ProcessGroup)] = []
        for i in range(len(num_gpus_list)):
            pg_ranks = ranks[: num_gpus_list[i]]
            process_groups.append(
                (dist.get_rank() in pg_ranks, dist.new_group(pg_ranks))
            )

        results: list[list[int]] = [
            [0] * len(process_groups) for _ in range(len(self.model.layers))
        ]
        # profile each layer with each process group
        for layer_index, layer in enumerate(self.model.layers):
            for pg_index, (should_run, pg) in enumerate(process_groups):
                if should_run:
                    results[layer_index][pg_index] = Profiler.profile_allreduce_layer(
                        layer, pg
                    )

        dist.barrier()
        # broadcast results to other nodes
        # 2d tensor, for each layer, multiple allreduce with different number of nodes
        results: torch.Tensor = torch.tensor(
            results, dtype=torch.float32, device="cuda", requires_grad=False
        )
        dist.broadcast(results, 0)

        return [
            {num_gpus_list[i]: result[i] for i in range(len(result))}
            for result in results.tolist()
        ]


def get_profile_path(model_tag: str, micro_batch_size: int, world_size: int, num_workers_per_node: int) -> Path:
    return Path(PROFILE_CACHE) / f"{model_tag}-{micro_batch_size}-{world_size}-{num_workers_per_node}"


def profile(
    args: OobleckArguments,
    master_addr: str,
    master_port: int,
    num_workers_per_node: int,
    world_size: int,
    rank: int,
):
    """Profile the given model and return a list of execution result
    per layer.
    ExecutionResult includes forward/backward latency, allreduce in node,
    and allreduce across nodes.

    Result is stored in cache for future use.
    Path: /workspace/Oobleck/tmp/profiles/{model_name}-{tag}/{layers|allreduce_in_node|allreduce_across_nodes}
    """
    directory = get_profile_path(args.model.model_tag, args.job.microbatch_size, world_size, num_workers_per_node)
    directory.mkdir(parents=True, exist_ok=True)

    logger.info("Profiling model %s. rank %d, world_size %d, num_workers_per_node %d.", args.model.model_name, rank, world_size, num_workers_per_node)


    dataset = OobleckDataset(
        args.model.model_name, args.model.dataset_path, args.model.dataset_name
    )
    logger.info("Dataset loaded")
    model = OobleckModel(
        args.model.model_name,
        dataset.sample,
        None,
        args.model.model_tag,
        args.model.model_args,
    )
    logger.info("model loaded")
    device = torch.device("cuda")
    for layer in model.layers:
        init_tensors(layer, device)

    profiler = Profiler(model, num_workers_per_node, world_size)

    assert not dist.is_initialized(), "Distributed is already initialized."
    logger.info(f"init pg. master_addr: {master_addr}, master_port: {master_port}")
    store = dist.TCPStore(
        host_name=master_addr,
        port=master_port,
        world_size=world_size,
        is_master=bool(rank == 0),
        wait_for_workers=False,
    )
    # 把rank中的所有GPU组成一个process group
    dist.init_process_group(
        backend="nccl", store=store, rank=rank, world_size=world_size
    )

    path = directory / f"mb{args.job.microbatch_size}.json"
    logger.info("Profiling model execution latency.")
    layer_execution_result = profiler.profile_execution_layers(args.job.microbatch_size)
    # Hack: 4 GPUs per node
    # In each node, the first process writes a file.
    if dist.get_rank() % 4 == 0:
        with path.open(mode="w") as f:
            json.dump(layer_execution_result, f)
            f.flush()

    path = directory / "allreduce_across_nodes.json"
    logger.info("Profiling cross-node allreduce latency.")
    allreduce_across_nodes = profiler.profile_allreduce_across_nodes()
    
    if dist.get_rank() % 4 == 0:
        with path.open(mode="w") as f:
            json.dump(allreduce_across_nodes, f)
            f.flush()

    path = directory / "allreduce_in_node.json"
    logger.info("Profiling in-node allreduce latency.")
    allreduce_in_node = profiler.profile_allreduce_in_node(num_workers_per_node)
    if dist.get_rank() % 4 == 0:
        with path.open(mode="w") as f:
            json.dump(allreduce_in_node, f)
            f.flush()

    # export configuration
    path = directory / "model_args.json"
    if dist.get_rank() % 4 == 0:
        with open(path, "w") as f:
            json.dump(args.model.model_args, f)

    dist.barrier()
    dist.destroy_process_group()
    assert not dist.is_initialized()


def validate_model_args(args: OobleckArguments) -> bool:
    directory = get_profile_path(args.model.model_name, args.model.model_tag)
    path = directory / "model_args.json"

    if not directory.exists() or not path.exists():
        return False

    args_same: bool = True
    with path.open() as f:
        args_from_file = json.load(f)
        for k, v in args.model.model_args.items():
            if k not in args_from_file or args_from_file[k] != v:
                args_same = False
                break
    return args_same