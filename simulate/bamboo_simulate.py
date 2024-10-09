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
from oobleck.csrc.planning.pipeline_template import (
    LayerExecutionResults,
    PipelineTemplate,
    PipelineTemplateGenerator,
    get_profile_results,
)
from oobleck.elastic.training_util import OobleckArguments
from oobleck.execution.dataloader import LoaderType, OobleckDataLoader
from oobleck.execution.dataset import OobleckDataset
from oobleck.execution.engine import ReconfigurationEngine
from oobleck.execution.pipeline import OobleckPipeline
from oobleck.module.model import OobleckModel
from oobleck.planning.instantiator import (
    HeterogeneousPipelinesExecutionPlan,
    PipelineInstantiator,
)
from tests.conftest import OobleckSingleProcessTestCase
from deepspeed.utils.logging import LoggerFactory, log_dist
from deepspeed.utils.logging import logging
import argparse
import json
import random
from argparse import Namespace
from oobleck.csrc.planning.pipeline_template import StageExecutionResult

logger = LoggerFactory.create_logger("bamboo_engine", logging.DEBUG)
CUDA_MEMORY = None


def create_pipeline_template_from_obj(obj) -> PipelineTemplate:
    stages = []
    num_layers = 0
    for json_stage in obj["stages"]:
        num_layers += len(json_stage["layer_indices"])
        stages.append(
            StageExecutionResult(json_stage["layer_indices"], json_stage["num_gpus"])
        )
    return PipelineTemplate(
        stages,
        obj["iteration_time"],
        num_layers,
        obj["num_nodes"],
        obj["num_gpus_per_node"],
    )


def create_pipeline_templates_from_json_file(file_path: str):
    obj = None
    with open(file_path, "r") as fp:
        obj = json.load(fp)
    json_pipeline_templates = obj["pipeline_template"]
    pipeline_templates: list[PipelineTemplate] = []
    for json_template in json_pipeline_templates:
        template = create_pipeline_template_from_obj(json_template)
        pipeline_templates.append(template)
    return pipeline_templates


def find_equal_pipeline_template(
    pipeline_templates: list[PipelineTemplate], json_pipeline
) -> PipelineTemplate:
    for pipeline_template in pipeline_templates:
        if (
            len(pipeline_template.get_stages()) != len(json_pipeline["stages"])
            or pipeline_template._num_nodes != json_pipeline["num_nodes"]
            or pipeline_template._num_gpus_per_node
            != json_pipeline["num_gpus_per_node"]
        ):
            continue
        is_same = True
        for i in range(len(json_pipeline["stages"])):
            stage = pipeline_template.get_stages()[i]
            json_stage = json_pipeline["stages"][i]
            layer_indices = stage._layer_indices
            json_layer_indices = json_stage["layer_indices"]
            if stage._num_gpus != json_stage["num_gpus"] or len(layer_indices) != len(
                json_layer_indices
            ):
                is_same = False
                break
            for j in range(len(layer_indices)):
                if layer_indices[j] != json_layer_indices[j]:
                    is_same = False
                    break
            if not is_same:
                break
        if is_same:
            return pipeline_template

    raise Exception("Need to be a template in pipeline templates")


def create_heterogeneous_plan_from_json_file(
    file_path: str, pipeline_templates: list[PipelineTemplate]
) -> HeterogeneousPipelinesExecutionPlan:
    obj = None
    with open(file_path, "r") as fp:
        obj = json.load(fp)
    json_execution_plan = obj["execution_plan"]

    plan_pipeline_templates = [
        find_equal_pipeline_template(pipeline_templates, json_pipeline)
        for json_pipeline in json_execution_plan["pipeline_templates"]
    ]

    num_instances_set = {
        find_equal_pipeline_template(pipeline_templates, kv[0]): kv[1]
        for kv in json_execution_plan["num_instances_set"]
    }

    num_microbatches_set = {
        find_equal_pipeline_template(pipeline_templates, kv[0]): kv[1]
        for kv in json_execution_plan["num_microbatches_set"]
    }

    return HeterogeneousPipelinesExecutionPlan(
        plan_pipeline_templates,
        num_instances_set,
        num_microbatches_set,
        json_execution_plan["allreduce_across_nodes"],
    )


class BambooEngine:
    def __init__(
        self,
        world_size: int,
        pp_size: int,
        microbatch: int,
        args: OobleckArguments,
    ):
        self._world_size = world_size
        self._pp_size = pp_size
        assert (
            self._world_size % self._pp_size == 0
        ), "Bamboo only supports Homogeneous pipeline"
        self._dp_size = self._world_size / self._pp_size
        self._microbatch = microbatch
        self._args = args
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
        dataset = OobleckDataset(
            self._args.model.model_name,
            self._args.model.dataset_path,
            self._args.model.dataset_name,
            (
                self._args.model.model_args["n_positions"]
                if "n_positions" in self._args.model.model_args
                else None
            ),
        )
        self._model = OobleckModel(
            self._args.model.model_name,
            dataset.sample,
            self._hf_training_args,
            self._args.model.model_tag,
            self._args.model.model_args,
        )

    def get_pipeline(self):
        # calculate per stage layer num
        layer_num = len(self._model.layers)
        base_layer_num_per_stage = layer_num // self._pp_size
        extra_layers = layer_num % self._pp_size
        layer_num_per_stage = [base_layer_num_per_stage] * self._pp_size
        for i in range(extra_layers):
            layer_num_per_stage[i] += 1

        # generate pipeline
        pipeline = []
        layer_id = 0
        for layer_num in layer_num_per_stage:
            sizes = []
            names = []
            for _i in range(layer_num):
                for name, param in self._model.layers[layer_id].named_parameters():
                    sizes.append(param.size())
                    names.append(name)
                layer_id += 1
            pipeline.append({"sizes": sizes, "names": names})

        return pipeline


def simulate_lost(
    model: str, microbatch: int, world_size: int, pp_size: int, out_dir: str
):
    print(
        f"simulate lost: {model}, mbs {world_size}, world_size {world_size}, pp_size {pp_size}"
    )
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
    engine = BambooEngine(
        world_size,
        pp_size,
        microbatch,
        args,
    )
    pipeline = engine.get_pipeline()
    # 模拟pipeline的复制

    pipeline_ranks = list(range(len(pipeline)))

    replicate_ranks = [rank + pp_size for rank in pipeline_ranks]

    result = {}
    result["world_size"] = pp_size * 2
    layers = []

    for idx, stage in enumerate(pipeline):
        obj = {}
        obj["sizes"] = stage["sizes"]
        obj["names"] = stage["names"]
        obj["ranks"] = [pipeline_ranks[idx], replicate_ranks[idx]]
        layers.append(obj)
    result["layers"] = layers

    with open(
        f"{out_dir}/{model}-{world_size}-{pp_size}-{microbatch}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bamboo simulator")
    parser.add_argument("--model", type=str, help="model tag")
    parser.add_argument("--microbatch", type=int, help="microbatch size")
    parser.add_argument("--worldsize", type=int, help="world size")
    parser.add_argument(
        "--pp", type=int, default=0, help="Pipeline parallel size"
    )
    parser.add_argument("--out-dir", type=str, help="output dir of result")
    parser.add_argument("--cuda-memory", type=int)

    args = parser.parse_args()
    

    CUDA_MEMORY = args.cuda_memory
    if CUDA_MEMORY is None:
        CUDA_MEMORY = torch.cuda.get_device_properties("cuda:0").total_memory
    simulate_lost(args.model, args.microbatch, args.worldsize, args.pp, args.out_dir)
