from __future__ import annotations
import copy
from typing import Mapping, Sequence
from transformers.training_args import TrainingArguments as HFTrainingArguments
import logging
from multiprocessing import connection
from unittest.mock import MagicMock
import math
import pytest
from pytest_mock import MockerFixture
import torch
from oobleck.csrc.planning.pipeline_template import LayerExecutionResults, PipelineTemplate, PipelineTemplateGenerator, get_profile_results
from oobleck.elastic.training_util import OobleckArguments
from oobleck.execution.dataloader import LoaderType, OobleckDataLoader
from oobleck.execution.dataset import OobleckDataset
from oobleck.execution.engine import ReconfigurationEngine
from oobleck.execution.pipeline import OobleckPipeline
from oobleck.module.model import OobleckModel
from oobleck.planning.instantiator import HeterogeneousPipelinesExecutionPlan, PipelineInstantiator
from tests.conftest import OobleckSingleProcessTestCase
from deepspeed.utils.logging import LoggerFactory, log_dist
from deepspeed.utils.logging import logging
import argparse
import json
import random
logger = LoggerFactory.create_logger("oobleck_engine", logging.DEBUG)
CUDA_MEMORY = None


class SimulatorEngine:
    def __init__(self, num_nodes, num_gpus_per_node, args: OobleckArguments, microbatch: int, world_size: int) -> None:
        args.job.microbatch_size = microbatch
        args.dist.world_size = world_size

        self._num_nodes: int = num_nodes
        self._num_gpus_per_node: int = num_gpus_per_node
        self._args = args
        self._dataset: OobleckDataset
        self._model: OobleckModel
        self._pipeline_templates: list[PipelineTemplate]
        training_args = {
            "output_dir": f"/workspace/Oobleck/tmp/output/{args.model.model_name}-{args.model.model_tag}",
            "per_device_train_batch_size": args.job.microbatch_size,
            "no_cuda": True,  # don't use cuda in HFTrainingArguments
            "log_level": "error",  # omit warning messages from HFTrainingArguments
            # do not set gradient_accumulation_steps in HFTrainingArguments
            "max_steps": args.job.steps,
        }
        self._hf_training_args: HFTrainingArguments = HFTrainingArguments(
            **training_args
        )
        (
            self._dataset,
            self._model,
            self._profile_results,
            self._pipeline_templates,
        ) = self._initialize_engine()
        global_num_microbatch = args.job.global_microbatch_size // args.job.microbatch_size
        self.instantiate_pipelines(global_num_microbatch)

    def _initialize_engine(
        self
    ) -> tuple[
        OobleckDataset,
        OobleckModel,
        LayerExecutionResults,
        list[PipelineTemplate],
    ]:
        dataset = OobleckDataset(
            self._args.model.model_name,
            self._args.model.dataset_path,
            self._args.model.dataset_name,
            self._args.model.model_args["n_positions"]
            if "n_positions" in self._args.model.model_args
            else None,
        )

        model = OobleckModel(
            self._args.model.model_name,
            dataset.sample,
            self._hf_training_args,
            self._args.model.model_tag,
            self._args.model.model_args,
        )
        profile_results: LayerExecutionResults = get_profile_results(
            self._args.model.model_tag,
            self._hf_training_args.per_device_train_batch_size,
            self._args.dist.world_size,
            self._args.dist.num_workers
        )
        
        total_memory_consumption = 6 * sum(
            [layer_result._mem_required[0] for layer_result in profile_results.get()]
        )
        total_memory_consumption += max(
            [layer_result._mem_required[1] for layer_result in profile_results.get()]
        )
        min_num_nodes = max(
            1,
            math.ceil(
                total_memory_consumption
                / (
                    CUDA_MEMORY
                    * self._num_gpus_per_node
                )
            ),
        )
        max_num_nodes = self._num_nodes
        assert min_num_nodes <= max_num_nodes, (
            "Minimum required number of nodes is larger than maximum number of nodes "
            f"(minimum required: {min_num_nodes}, you have: {max_num_nodes})."
        )

        logger.info(f"Number of nodes range: ({min_num_nodes}, {max_num_nodes})")

        # TODO: Calculate num_gpus_range based on profile results
        print(f"min_nodes: {min_num_nodes}, max_nodes: {max_num_nodes}, gpus: {self._num_gpus_per_node}, ")
        template_generator = PipelineTemplateGenerator()
        # pipeline_templates: list[
        #     PipelineTemplate
        # ] = template_generator.create_pipeline_templates_serial(
        #     profile_results, (min_num_nodes, max_num_nodes), self._num_gpus_per_node
        # )
        pipeline_templates: list[
            PipelineTemplate
        ] = template_generator.create_pipeline_templates(
            profile_results, (min_num_nodes, max_num_nodes), num_gpus_per_node
        )
        return dataset, model, profile_results, pipeline_templates
    

    def instantiate_pipelines(self, global_num_microbatch: int):
        # 这里获取最佳的pipeline
        instantiator = PipelineInstantiator()
        execution_plan: HeterogeneousPipelinesExecutionPlan = (
            instantiator.get_best_execution_plan(
                self._pipeline_templates,
                [
                    layer._allreduce_across_nodes
                    for layer in self._profile_results.get()
                ],
                self._num_nodes,
                global_num_microbatch,
            )
        )

        # TODO: get current iteration progress
        dataloader: OobleckDataLoader = OobleckDataLoader(
            args=self._hf_training_args,
            datasets=self._dataset,
            dataloader_type=LoaderType.Training,
            pipeline_index = 0,
            num_microbatches=execution_plan.num_microbatches,
            num_iterations_done=0,
            epoch=0,
        )

        self._pipeline: OobleckPipeline
        # pipelines是最佳的所有的pipeline的组合。self._pipeline是本rank所在的pipeline
        self._pipeline, self._pipelines = execution_plan.instantiate_simulated(
            model=self._model,
            dataloader=dataloader,
            training_args=self._hf_training_args,
            num_gpus_per_node=self._num_gpus_per_node,
            step=0,
        )
        self.num_microbatches = copy.deepcopy(execution_plan.num_microbatches)
        # rank_grids: layer -> [ranks]
        self.old_rank_grids =  [
            copy.deepcopy(pipeline.rank_grid) for pipeline in self._pipelines
        ]
        

    def get_pipelines(self):
        num_pipeline = len(self.old_rank_grids)
        pipelines = []
        for i in range(num_pipeline):
            num_layers_per_stage = []
            layers = 1
            ranks = [rank[0] for (layer, rank) in self.old_rank_grids[i].items()]
            curr_rank = ranks[0]
            
            for j in range(1, len(ranks)):
                if curr_rank == ranks[j]:
                    layers += 1
                else:
                    num_layers_per_stage.append(layers)
                    layers = 1
                    curr_rank = ranks[j]
            num_layers_per_stage.append(layers)
            assert(sum(num_layers_per_stage) == len(self.old_rank_grids[i]))
            pipeline = {"layers": self.old_rank_grids[i], "num_of_microbatches": self.num_microbatches[i], "num_layers_per_stage": num_layers_per_stage}
            pipelines.append(pipeline)
        return pipelines

def simulate_lost(model: str, microbatch: int, world_size: int, lost_nodes: int, out_dir: str): 
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
    engine = SimulatorEngine(world_size, 1, args, microbatch, world_size)
    pipelines = engine.get_pipelines()

    # 模拟要丢失的nodes
    assert(len(pipelines) > 1)
    can_lost_pipeline_stages = sum([len(pipeline["num_layers_per_stage"]) for pipeline in pipelines[: -1]])
    assert(can_lost_pipeline_stages >= lost_nodes)

    ignore_Embedding = True
    if lost_nodes > can_lost_pipeline_stages - 2 * (len(pipelines) - 1):
        ignore_Embedding = False

    
    lost_ranks:list[int] = []
    
    pipeline_id = 0
    # 已知：每个rank只能有一个pipeline stage
    while len(lost_ranks) < lost_nodes:
        pipeline_stages = pipelines[pipeline_id]["num_layers_per_stage"]
        if ignore_Embedding and len(pipeline_stages) > 1:
            lost_stage_id = 1
        else:
            lost_stage_id = 0

        # 尝试找到一个可以丢失的rank
        while True:
            if lost_stage_id == len(pipeline_stages):
                break
            lost_layer_id = sum(pipeline_stages[:lost_stage_id])
            # print(f"lost_stage_id: {lost_stage_id}, lost_layer_id: {lost_layer_id}")
            lost_rank = pipelines[pipeline_id]["layers"][lost_layer_id][0]
            if lost_rank in lost_ranks:
                lost_stage_id += 1
            else:
                lost_ranks.append(lost_rank)
                print(f"lost_rank: {lost_rank}. lost_pipeline_id: {pipeline_id}, lost_stage_id: {lost_stage_id}, lost_layer_id: {lost_layer_id}")
                break
        pipeline_id = (pipeline_id + 1) % (len(pipelines) - 1) 
    lost_ranks.sort()
    print(f"lost ranks: {lost_ranks}")


    reconfigure_engine = ReconfigurationEngine(engine, engine._pipelines, is_simulated=True)
    reconfigure_engine.on_reconfigure(lost_ranks)
    print(f"after reconfigure. old_grid_ranks: {reconfigure_engine._record_old_rank_grids}, new_grid_ranks: {reconfigure_engine._record_new_rank_grids}")
    result = get_reconfigure_layers(engine, reconfigure_engine._record_old_rank_grids, reconfigure_engine._record_new_rank_grids, lost_ranks)
    result["world_size"] = world_size
    with open(f'{out_dir}/{model}-{world_size}-{microbatch}-{lost_nodes}.json', 'w', encoding='utf-8') as f:
        json.dump(result,f)
    # print(result)


def get_reconfigure_layers(engine: SimulatorEngine, old_rank_grids: Sequence[Mapping[int, Sequence[int]]], new_rank_grids: Sequence[Mapping[int, Sequence[int]]], lost_ranks: list[int]):
    # layer -> list of ranks contains that layer
    old_layer_map:dict[int, list[list[int]]] = {}
    for pipeline in old_rank_grids:
        for layer, ranks in pipeline.items():
            if layer in old_layer_map:
                for i, rank in enumerate(ranks):
                    old_layer_map[layer][i].append(rank)
            else:
                old_layer_map[layer] = [ranks]
    # layer -> list of ranks container that layer
    new_layer_map:dict[int, list[list[int]]] = {}
    for pipeline in new_rank_grids:
        for layer, ranks in pipeline.items():
            if layer in new_layer_map:
                for i, rank in enumerate(ranks):
                    new_layer_map[layer][i].append(rank)
            else:
                new_layer_map[layer] = [ranks]

    print(f"old_layer_map: {old_layer_map}")
    print(f"new_layer_map: {new_layer_map}")
    assert(len(old_layer_map) == len(new_layer_map))

    # layer -> process group of broadcast (the first rank in the pg will broadcast to others)
    reconfigure_layers : dict[int, list[int]]= {}
    for layer in old_layer_map.keys():
        old_ranks: set[int] = {rank for fsdp_ranks in old_layer_map[layer] for rank in fsdp_ranks}
        new_ranks: set[int] = {rank for fsdp_ranks in new_layer_map[layer] for rank in fsdp_ranks} 
        unite_ranks = old_ranks.intersection(new_ranks)
        diff_ranks = new_ranks.difference(unite_ranks)
        if len(diff_ranks) > 0:
            if len(unite_ranks) > 0:
                reconfigure_layers[layer] = [next(iter(unite_ranks))] + list(diff_ranks)
            elif len(old_ranks.difference(lost_ranks)) > 0:
                reconfigure_layers[layer] = [next(iter(old_ranks.difference(lost_ranks)))] + list(diff_ranks)
            else:
                raise Exception("Can not reconfigure. No redudant data")

    print(f"original reconfigure_layers: {reconfigure_layers}")
    # map candidiate ranks to new serial numbers
    # candidate_ranks = {rank for ranks in reconfigure_layers.values() for rank in ranks}
    # rank_map:dict[int, int] = {}
    # new_rank_id = 0
    # for candidate_rank in candidate_ranks:
    #     rank_map[candidate_rank] = new_rank_id
    #     new_rank_id += 1
    # for ranks in reconfigure_layers.values():
    #     for i in range(len(ranks)):
    #         ranks[i] = rank_map[ranks[i]]

    # print(f"reconfigure_layers: {reconfigure_layers}")
    result = []
    for layer_id, ranks in  reconfigure_layers.items():
        sizes = []
        names = []
        p = 0
        for name, param in engine._model.layers[layer_id].named_parameters():
            sizes.append(param.size())
            names.append(name)
            p += param.numel()
        print(f"parameters: {p}")
        result.append({"sizes": sizes, "ranks": ranks, "names": names})
    return {"layers": result}

def simulate_pipelines(model: str, microbatch: int, world_size: int):
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
    engine = SimulatorEngine(world_size, 1, args, microbatch, world_size)
    pipelines = engine.get_pipelines()
    result = {"model": model, "microbatch_size": microbatch, "world_size": world_size, "pipelines": pipelines}
    with open(f'/workspace/Oobleck/tmp/pipelines/{model}-{microbatch}-{world_size}.json', 'w', encoding='utf-8') as f:
        json.dump(result,f)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Oobleck simulator')
    parser.add_argument('--model', type=str, help='model tag')
    parser.add_argument('--microbatch', type=int, help='microbatch size')
    parser.add_argument('--worldsize', type=int, help='world size')
    parser.add_argument('--lost-nodes', type=int, default=0, help='The number of nodes want to lost')
    parser.add_argument('--out-dir', type=str, help='output dir of result')
    parser.add_argument('--cuda-memory', type=int)
    args = parser.parse_args()

    CUDA_MEMORY = args.cuda_memory
    if args.lost_nodes == 0:
        simulate_pipelines(args.model, args.microbatch, args.worldsize)
    else:
        simulate_lost(args.model, args.microbatch, args.worldsize, args.lost_nodes, args.out_dir)









# old_rank_grids:[
#     {0: [0], 1: [0], 2: [0], 3: [0], 4: [0], 5: [0], 6: [0], 7: [0], 8: [0], 9: [0], 10: [0], 11: [0], 12: [0], 13: [0], 14: [0], 15: [0], 16: [0], 17: [0], 18: [0], 19: [0], 20: [0], 21: [0], 22: [0], 23: [0], 24: [0], 25: [0], 26: [0], 27: [0], 28: [0], 29: [0], 30: [0], 31: [0], 32: [1], 33: [1], 34: [1], 35: [1], 36: [1], 37: [1], 38: [1], 39: [1], 40: [1], 41: [1], 42: [1], 43: [1], 44: [1], 45: [1], 46: [1], 47: [1], 48: [1], 49: [1], 50: [1], 51: [1], 52: [1], 53: [1], 54: [1], 55: [1], 56: [1], 57: [1], 58: [1], 59: [1], 60: [1], 61: [1]},
#     {0: [2], 1: [2], 2: [2], 3: [2], 4: [2], 5: [2], 6: [2], 7: [2], 8: [2], 9: [2], 10: [2], 11: [2], 12: [2], 13: [2], 14: [2], 15: [2], 16: [2], 17: [2], 18: [2], 19: [2], 20: [2], 21: [2], 22: [3], 23: [3], 24: [3], 25: [3], 26: [3], 27: [3], 28: [3], 29: [3], 30: [3], 31: [3], 32: [3], 33: [3], 34: [3], 35: [3], 36: [3], 37: [3], 38: [3], 39: [3], 40: [3], 41: [3], 42: [4], 43: [4], 44: [4], 45: [4], 46: [4], 47: [4], 48: [4], 49: [4], 50: [4], 51: [4], 52: [4], 53: [4], 54: [4], 55: [4], 56: [4], 57: [4], 58: [4], 59: [4], 60: [4], 61: [4]}
#     ]

# new_rank_grids: [
#     {0: [0], 1: [0], 2: [0], 3: [0], 4: [0], 5: [0], 6: [0], 7: [0], 8: [0], 9: [0], 10: [0], 11: [0], 12: [0], 13: [0], 14: [0], 15: [0], 16: [0], 17: [0], 18: [0], 19: [0], 20: [0], 21: [0], 22: [0], 23: [0], 24: [0], 25: [0], 26: [0], 27: [0], 28: [0], 29: [0], 30: [0], 31: [0], 32: [1], 33: [1], 34: [1], 35: [1], 36: [1], 37: [1], 38: [1], 39: [1], 40: [1], 41: [1], 42: [1], 43: [1], 44: [1], 45: [1], 46: [1], 47: [1], 48: [1], 49: [1], 50: [1], 51: [1], 52: [1], 53: [1], 54: [1], 55: [1], 56: [1], 57: [1], 58: [1], 59: [1], 60: [1], 61: [1]},
#     {0: [2], 1: [2], 2: [2], 3: [2], 4: [2], 5: [2], 6: [2], 7: [2], 8: [2], 9: [2], 10: [2], 11: [2], 12: [2], 13: [2], 14: [2], 15: [2], 16: [2], 17: [2], 18: [2], 19: [2], 20: [2], 21: [2], 22: [2], 23: [2], 24: [2], 25: [2], 26: [2], 27: [2], 28: [2], 29: [2], 30: [2], 31: [2], 32: [3], 33: [3], 34: [3], 35: [3], 36: [3], 37: [3], 38: [3], 39: [3], 40: [3], 41: [3], 42: [3], 43: [3], 44: [3], 45: [3], 46: [3], 47: [3], 48: [3], 49: [3], 50: [3], 51: [3], 52: [3], 53: [3], 54: [3], 55: [3], 56: [3], 57: [3], 58: [3], 59: [3], 60: [3], 61: [3]}
# ]
def current_not_used():
    old_rank_grids = [
        {0: [0], 1: [0], 2: [0], 3: [0], 4: [0], 5: [0], 6: [0], 7: [0], 8: [0], 9: [0], 10: [0], 11: [0], 12: [0], 13: [0], 14: [0], 15: [0], 16: [0], 17: [0], 18: [0], 19: [0], 20: [0], 21: [0], 22: [0], 23: [0], 24: [0], 25: [0], 26: [0], 27: [0], 28: [0], 29: [0], 30: [0], 31: [0], 32: [1], 33: [1], 34: [1], 35: [1], 36: [1], 37: [1], 38: [1], 39: [1], 40: [1], 41: [1], 42: [1], 43: [1], 44: [1], 45: [1], 46: [1], 47: [1], 48: [1], 49: [1], 50: [1], 51: [1], 52: [1], 53: [1], 54: [1], 55: [1], 56: [1], 57: [1], 58: [1], 59: [1], 60: [1], 61: [1]},
        {0: [2], 1: [2], 2: [2], 3: [2], 4: [2], 5: [2], 6: [2], 7: [2], 8: [2], 9: [2], 10: [2], 11: [2], 12: [2], 13: [2], 14: [2], 15: [2], 16: [2], 17: [2], 18: [2], 19: [2], 20: [2], 21: [2], 22: [3], 23: [3], 24: [3], 25: [3], 26: [3], 27: [3], 28: [3], 29: [3], 30: [3], 31: [3], 32: [3], 33: [3], 34: [3], 35: [3], 36: [3], 37: [3], 38: [3], 39: [3], 40: [3], 41: [3], 42: [4], 43: [4], 44: [4], 45: [4], 46: [4], 47: [4], 48: [4], 49: [4], 50: [4], 51: [4], 52: [4], 53: [4], 54: [4], 55: [4], 56: [4], 57: [4], 58: [4], 59: [4], 60: [4], 61: [4]}
        ]
    new_rank_grids = [
        {0: [0], 1: [0], 2: [0], 3: [0], 4: [0], 5: [0], 6: [0], 7: [0], 8: [0], 9: [0], 10: [0], 11: [0], 12: [0], 13: [0], 14: [0], 15: [0], 16: [0], 17: [0], 18: [0], 19: [0], 20: [0], 21: [0], 22: [0], 23: [0], 24: [0], 25: [0], 26: [0], 27: [0], 28: [0], 29: [0], 30: [0], 31: [0], 32: [1], 33: [1], 34: [1], 35: [1], 36: [1], 37: [1], 38: [1], 39: [1], 40: [1], 41: [1], 42: [1], 43: [1], 44: [1], 45: [1], 46: [1], 47: [1], 48: [1], 49: [1], 50: [1], 51: [1], 52: [1], 53: [1], 54: [1], 55: [1], 56: [1], 57: [1], 58: [1], 59: [1], 60: [1], 61: [1]},
        {0: [2], 1: [2], 2: [2], 3: [2], 4: [2], 5: [2], 6: [2], 7: [2], 8: [2], 9: [2], 10: [2], 11: [2], 12: [2], 13: [2], 14: [2], 15: [2], 16: [2], 17: [2], 18: [2], 19: [2], 20: [2], 21: [2], 22: [2], 23: [2], 24: [2], 25: [2], 26: [2], 27: [2], 28: [2], 29: [2], 30: [2], 31: [2], 32: [3], 33: [3], 34: [3], 35: [3], 36: [3], 37: [3], 38: [3], 39: [3], 40: [3], 41: [3], 42: [3], 43: [3], 44: [3], 45: [3], 46: [3], 47: [3], 48: [3], 49: [3], 50: [3], 51: [3], 52: [3], 53: [3], 54: [3], 55: [3], 56: [3], 57: [3], 58: [3], 59: [3], 60: [3], 61: [3]}
    ]
    # layer->[fsdp index[GPUs]]
    old_layer_map:dict[int, list[list[int]]] = {}
    for pipeline in old_rank_grids:
        for layer, ranks in pipeline.items():
            if layer in old_layer_map:
                for i, rank in enumerate(ranks):
                    old_layer_map[layer][i].append(rank)
            else:
                old_layer_map[layer] = [ranks]


    new_layer_map:dict[int, list[list[int]]] = {}
    for pipeline in new_rank_grids:
        for layer, ranks in pipeline.items():
            if layer in new_layer_map:
                for i, rank in enumerate(ranks):
                    new_layer_map[layer][i].append(rank)
            else:
                new_layer_map[layer] = [ranks]

    fsdp_size = len(new_layer_map[0])
    print(f"old_layer_map: {old_layer_map}")
    print(f"new_layer_map: {new_layer_map}")
    layer_nums = len(new_layer_map)
    assert(len(old_layer_map) == len(new_layer_map))

    # (layer, fsdp_index) -> ranks need broadcast
    reconfigure_layers : dict[tuple[int, int], set[int]]= {}

    for layer in old_layer_map.keys():
        for fsdp_index in range(len(old_layer_map[layer])):
            old_ranks = set(old_layer_map[layer][fsdp_index])
            new_ranks = set(new_layer_map[layer][fsdp_index])
            unite_ranks = old_ranks.intersection(new_ranks)
            assert(len(unite_ranks) > 0)
            diff_ranks = new_ranks.difference(unite_ranks)
            if len(diff_ranks) > 0:
                reconfigure_layers[(layer, fsdp_index)] = diff_ranks

    print(f"reconfigure_layers: {reconfigure_layers}")

    # broadcast_time = [0, 10, 20, 30]
    def get_layer_size(layer_index: int, layer_nums: int):
        if layer_index == 0:
            return 0
        elif layer_index == layer_nums - 1:
            return 1 
        else :
            return 2
        
    # {(layer_size, broad_cast_nums) -> time}
    # 需要实测这个
    broadcast_time:dict[(int, int), float] = {}
    total_time = 0
    for (layer, fsdp_index), ranks in reconfigure_layers.items():
        total_time += broadcast_time[(get_layer_size(layer,), len(ranks))]
    print(f"total_time: {total_time}")