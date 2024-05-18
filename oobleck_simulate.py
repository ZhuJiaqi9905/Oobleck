from __future__ import annotations
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

logger = LoggerFactory.create_logger("oobleck_engine", logging.DEBUG)


class SimulatorEngine:
    def __init__(self, num_nodes, num_gpus_per_node, args: OobleckArguments) -> None:
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
            self._args.model.model_name,
            self._args.model.model_tag,
            self._hf_training_args.per_device_train_batch_size,
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
                    torch.cuda.get_device_properties("cuda:0").total_memory
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
        pipeline_templates: list[
            PipelineTemplate
        ] = template_generator.create_pipeline_templates_serial(
            profile_results, (min_num_nodes, max_num_nodes), self._num_gpus_per_node
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

def simulate(): 
    lose_ranks = [1]

    num_nodes = 5
    num_gpus_per_node = 1
    config_path = "./examples/simulate_gpt2.yaml"
    args = OobleckArguments.load_yaml(config_path)
    print(f"{args}")
    engine = SimulatorEngine(num_nodes, num_gpus_per_node, args)
    print("simulatorEngine init")
    reconfigure_engine = ReconfigurationEngine(engine, engine._pipelines, is_simulated=True)
    reconfigure_engine.on_reconfigure(lose_ranks)
    print(f"after reconfigure. old_grid_ranks: {reconfigure_engine._record_old_rank_grids}, new_grid_ranks: {reconfigure_engine._record_new_rank_grids}")

# old_rank_grids:[
#     {0: [0], 1: [0], 2: [0], 3: [0], 4: [0], 5: [0], 6: [0], 7: [0], 8: [0], 9: [0], 10: [0], 11: [0], 12: [0], 13: [0], 14: [0], 15: [0], 16: [0], 17: [0], 18: [0], 19: [0], 20: [0], 21: [0], 22: [0], 23: [0], 24: [0], 25: [0], 26: [0], 27: [0], 28: [0], 29: [0], 30: [0], 31: [0], 32: [1], 33: [1], 34: [1], 35: [1], 36: [1], 37: [1], 38: [1], 39: [1], 40: [1], 41: [1], 42: [1], 43: [1], 44: [1], 45: [1], 46: [1], 47: [1], 48: [1], 49: [1], 50: [1], 51: [1], 52: [1], 53: [1], 54: [1], 55: [1], 56: [1], 57: [1], 58: [1], 59: [1], 60: [1], 61: [1]},
#     {0: [2], 1: [2], 2: [2], 3: [2], 4: [2], 5: [2], 6: [2], 7: [2], 8: [2], 9: [2], 10: [2], 11: [2], 12: [2], 13: [2], 14: [2], 15: [2], 16: [2], 17: [2], 18: [2], 19: [2], 20: [2], 21: [2], 22: [3], 23: [3], 24: [3], 25: [3], 26: [3], 27: [3], 28: [3], 29: [3], 30: [3], 31: [3], 32: [3], 33: [3], 34: [3], 35: [3], 36: [3], 37: [3], 38: [3], 39: [3], 40: [3], 41: [3], 42: [4], 43: [4], 44: [4], 45: [4], 46: [4], 47: [4], 48: [4], 49: [4], 50: [4], 51: [4], 52: [4], 53: [4], 54: [4], 55: [4], 56: [4], 57: [4], 58: [4], 59: [4], 60: [4], 61: [4]}
#     ]

# new_rank_grids: [
#     {0: [0], 1: [0], 2: [0], 3: [0], 4: [0], 5: [0], 6: [0], 7: [0], 8: [0], 9: [0], 10: [0], 11: [0], 12: [0], 13: [0], 14: [0], 15: [0], 16: [0], 17: [0], 18: [0], 19: [0], 20: [0], 21: [0], 22: [0], 23: [0], 24: [0], 25: [0], 26: [0], 27: [0], 28: [0], 29: [0], 30: [0], 31: [0], 32: [1], 33: [1], 34: [1], 35: [1], 36: [1], 37: [1], 38: [1], 39: [1], 40: [1], 41: [1], 42: [1], 43: [1], 44: [1], 45: [1], 46: [1], 47: [1], 48: [1], 49: [1], 50: [1], 51: [1], 52: [1], 53: [1], 54: [1], 55: [1], 56: [1], 57: [1], 58: [1], 59: [1], 60: [1], 61: [1]},
#     {0: [2], 1: [2], 2: [2], 3: [2], 4: [2], 5: [2], 6: [2], 7: [2], 8: [2], 9: [2], 10: [2], 11: [2], 12: [2], 13: [2], 14: [2], 15: [2], 16: [2], 17: [2], 18: [2], 19: [2], 20: [2], 21: [2], 22: [2], 23: [2], 24: [2], 25: [2], 26: [2], 27: [2], 28: [2], 29: [2], 30: [2], 31: [2], 32: [3], 33: [3], 34: [3], 35: [3], 36: [3], 37: [3], 38: [3], 39: [3], 40: [3], 41: [3], 42: [3], 43: [3], 44: [3], 45: [3], 46: [3], 47: [3], 48: [3], 49: [3], 50: [3], 51: [3], 52: [3], 53: [3], 54: [3], 55: [3], 56: [3], 57: [3], 58: [3], 59: [3], 60: [3], 61: [3]}
# ]

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