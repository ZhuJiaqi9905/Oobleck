import random
from typing import Any, Optional, Type

import torch
from accelerate import init_empty_weights
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForPreTraining,
    PretrainedConfig,
    PreTrainedModel,
    TrainingArguments,
)

from oobleck.module.sharding import get_split_points, shard_model

RANDOM_SEED = 42

# Oobleck has been tested only with the following models.
lang_models = ["gpt2", "t5", "bert", "bloom"]
image_models = ["vit", "resnet", "clip", "swin"]

automodel_dict = {
    "gpt2": AutoModelForPreTraining,
    "t5": AutoModelForPreTraining,
    "bert": AutoModelForCausalLM,
    "bloom": AutoModelForPreTraining,
    "vit": AutoModelForImageClassification,
    "resnet": AutoModelForImageClassification,
    "clip": AutoModelForImageClassification,
    "swin": AutoModelForImageClassification,
}


class OobleckModel:
    """
    A wrapper model class of Hugging Face model
    downloaded from Hugging Face Hub (https://huggingface.co/models).

    It runs huggingface.utils.fx.symbolic_trace to get GraphModule
    and shard it to multiple GraphModules for pipeline execution.

    Model initialization must be done before distributed initialization.
    """

    def __init__(
        self,
        model_name: str,
        sample_inputs: dict[str, Any],
        training_args: Optional[TrainingArguments] = None,
        model_tag: Optional[str] = None,
        config_args: Optional[dict[str, Any]] = None,
    ):
        # Initialize CPU seed
        random.seed(RANDOM_SEED)
        torch.default_generator.manual_seed(RANDOM_SEED)

        if config_args is None:
            config_args = {}
        config_args["use_cache"] = False
        config_args["remove_unused_columns"] = False
        # necessary to register backward hooks
        config_args["return_dict"] = False
        # Use training_args for fp16/bf16
        if model_tag == "gpt3_2_7B":
            model_config =  AutoConfig.from_pretrained("/workspace/Oobleck/data/model/gpt3_2_7B/config.json")
        elif model_tag == "gpt3_1_3B":
            model_config = AutoConfig.from_pretrained("/workspace/Oobleck/data/model/gpt3_1_3B/config.json")
        elif model_tag == "gpt3_6_7B":
            model_config = AutoConfig.from_pretrained("/workspace/Oobleck/data/model/gpt3_6_7B/config.json")
        elif model_tag == "gpt3_350M":
            model_config = AutoConfig.from_pretrained("/workspace/Oobleck/data/model/gpt3_350M/config.json")
        elif model_tag == "bert_340M":
            model_config = AutoConfig.from_pretrained("/workspace/Oobleck/data/model/bert_340M/config.json")
        else:
            raise Exception(f"No model config for model: {model_tag}") 
        # model_config: PretrainedConfig = AutoConfig.from_pretrained(
        #     model_name, **config_args
        # )
        print(f"model_name: {model_name}, model_tag: {model_tag}")
        model: Optional[Type[PreTrainedModel]] = None
        with init_empty_weights():
            for key, automodel in automodel_dict.items():
                if key in model_name:
                    model = automodel.from_config(model_config, torch_dtype=torch.float16)
                    break
        assert model, f"Given model {model_name} is not supported yet."
        
        self.sample_inputs = sample_inputs
        self.trace_input_names = list(sample_inputs.keys())
        print(f"trace_input_names: {self.trace_input_names}")
        split_points = get_split_points(model_config)
        self.layers = shard_model(model, self.trace_input_names, split_points)
        self.model_name = model_name
        self.model_tag = model_tag

        # for i, graph_module in enumerate(self.layers):
        #     print(f"Graph Module {i+1}:")
        #     # 获取模型的图表示
        #     graph = graph_module.graph
            
        #     # 打印每个节点的详细信息
        #     for node in graph.nodes:
        #         print(f"Node name: {node.name}")
        #         print(f"  Target: {node.target}")
        #         print(f"  Args: {node.args}")
        #         print(f" opcode: {node.op}")

        #     # 可以根据需要打印其他信息，比如参数的尺寸
        #     for name, param in graph_module.named_parameters():
        #         print(f"Parameter name: {name}, Size: {param.size()}, dtype: {param.dtype}")
        #     print()
        #     print(f"code {graph_module.code}")

        self.total_num_params = sum(
            sum(p.numel() for p in layer.parameters()) for layer in self.layers
        )
        self.training_args = training_args
        self.model_args = model_config
