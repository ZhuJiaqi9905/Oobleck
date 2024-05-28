import json
import operator
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageClassification,
    AutoModelForPreTraining,
    PretrainedConfig,
    PreTrainedModel,
    TrainingArguments,
)
import torch
import torch.fx
from itertools import chain
from collections import defaultdict
from torch.fx.node import Node
from transformers.utils.fx import symbolic_trace
from typing import Type, List, Dict, Optional, Union, Any, Tuple
from transformers import PretrainedConfig


def get_split_points(config: Type[PretrainedConfig]) -> List[str]:
    split_points = []

    if "gpt" in config.model_type:
        for i in range(config.num_hidden_layers):
            split_points.append(f"transformer.h.{i}")
        split_points.append("transformer.ln_f")
    elif "bert" in config.model_type:
        for i in range(config.num_hidden_layers):
            split_points.append(f"bert.encoder.layer.{i}")
        split_points.append("cls")
    elif "t5" in config.model_type:
        for i in range(config.num_layers):
            split_points.append(f"encoder.block.{i}")
        for i in range(config.num_decoder_layers):
            split_points.append(f"decoder.block.{i}")
        split_points.append("lm_head")
    # Sharding for the Google's HuggingFace ViT model
    # e.g. google/vit-base-patch16-224 (https://huggingface.co/google/vit-base-patch16-224)
    elif "vit" in config.model_type:
        for i in range(config.num_hidden_layers):
            split_points.append(f"vit.encoder.layer.{i}")
        split_points.append("vit.layernorm")
    # Sharding for the Microsoft's HuggingFace ResNet model
    # e.g. microsoft/resnet-152 (https://huggingface.co/microsoft/resnet-152)
    elif "resnet" in config.model_type:
        for i, depth in enumerate(config.depths):
            for j in range(depth):
                split_points.append(f"resnet.encoder.stages.{i}.layers.{j}")
        split_points.append("resnet.pooler")

    assert (
        split_points
    ), f"Split points is empty. Check your model {config.model_type} is supported."

    return split_points


def _split_nodes(
    traced: torch.fx.GraphModule, split_points: List[str]
) -> Tuple[Dict[str, int], Dict[int, List[str]], Dict[str, int]]:
    """Analyze the given traced module and split it to subgraphs.
    While partitioning, it also finds additioanl required inputs and outputs
    so that they are added.

    Args:
        traced (torch.fx.GraphModule): A traced graph module to be split.
    """

    node_name_to_shard_id: Dict[str, int] = {}
    shard_id_to_node: Dict[int, List[Node]] = defaultdict(list)
    shard_id = 0

    nodes_so_far: List[str] = []
    extra_output: Dict[int, List[str]] = {}

    for node in traced.graph.nodes:
        if node.op == "placeholder":
            node_name_to_shard_id[node.name] = shard_id
            nodes_so_far.append(node.name)
            shard_id_to_node[shard_id].append(node)
        elif node.op in [
            "get_attr",
            "call_function",
            "call_method",
            "call_module",
        ]:
            node_name_to_shard_id[node.name] = shard_id
            nodes_so_far.append(node.name)
            shard_id_to_node[shard_id].append(node)

            point = next(
                filter(lambda p: node.next.name.startswith(p), split_points), None
            )
            if point:
                # Record outputs that should be used later, so that it can be added
                # in return of this shard
                outputs = []
                nodes = list(chain(*shard_id_to_node.values()))
                for node in nodes:
                    for user in node.users.keys():
                        if user.name not in node_name_to_shard_id:
                            outputs.append(node.name)

                extra_output[shard_id] = list(dict.fromkeys(outputs).keys())

                # If the current node is in the next shard, we increase shard count.
                shard_id += 1
                split_points.remove(point)

        elif node.op == "output":
            break

    assert len(split_points) == 0, "Sharding is not complete."

    return node_name_to_shard_id, extra_output


def shard_model(
    model: torch.nn.Module, concrete_args: List[str], split_points: List[str]
) -> List[torch.fx.GraphModule]:
    """Use torch.fx to do symbolic trace on the given model, and shard it to several subgraphs
    based on the given split_points.

    Code reference:
    1. https://github.com/HPDL-Group/Merak/blob/e8a2a779fea878be9b778f8a808a192364766f36/Merak/autoshard/graph_shard.py
    2. https://github.com/facebookresearch/fairscale/blob/5b38de380e4407c2ef02f357ebc640f53470ea24/fairscale/experimental/nn/auto_shard.py

    Args:
        model (torch.nn.Module): The model to be sharded.
        concrete_args (List[str]): Arguments that are used for symbolic_trace.
            This will be the list of inputs of the generated :class:`torch.fx.GraphModule`.

        split_points (List[str]): Module names that are split.

    Returns:
        List[torch.fx.GraphModule]: The list of sharded :class:`torch.fx.GraphModule`s.
    """
    module_list: List[torch.fx.GraphModule] = [] # 存储分片后的子图列表

    traced = symbolic_trace(model, input_names=concrete_args)
    split_points = [p.replace(".", "_") for p in split_points]
    # 使用分割点将图分片
    node_name_to_shard_id, extra_outputs = _split_nodes(traced, split_points)

    prev_shard_id = 1000
    prev_node: Optional[Node] = None

    env: Dict[str, Node] = {}
    prev_node: Optional[Node] = None

    new_graph = torch.fx.Graph() # 创建一个新图用于存储分片后的子图
    # Iterate all nodes
    for node in traced.graph.nodes:
        if node.name in node_name_to_shard_id:
            current_shard_id = node_name_to_shard_id[node.name]
            if prev_shard_id < current_shard_id:
                assert prev_node, "prev_node cannot be None"

                # If the current node is in the next shard, we insert an output node.
                # A new graph is created an a placeholder is added for the next shard.
                # 如果当前节点在下一个分片中，则插入一个输出节点
                # 创建一个新图，并为下一个分片添加一个占位符
                with new_graph.inserting_after(prev_node):
                    if prev_shard_id in extra_outputs:
                        outputs = extra_outputs[prev_shard_id]
                        outputs = tuple([env[i] for i in outputs])
                        new_graph.output(outputs)
                    else:
                        new_graph.output(tuple(env[prev_node.name]))

                new_graph.lint()
                module_list.append(torch.fx.GraphModule(model, new_graph))

                # Create a new graph
                new_graph = torch.fx.Graph()
                for output in outputs:
                    # Add all nodes in return of the previous graph to its input
                    node_name = env[output.name].name
                    pl_node = new_graph.create_node("placeholder", node_name)
                    env[node_name] = pl_node

        # Cut is done. Add all nodes into the current graph (except for labels placeholder).
        if node.op in [
            "placeholder",
            "get_attr",
            "call_function",
            "call_method",
            "call_module",
        ]:
            # Copy the nodes from the existing graph to the new graph.
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node
        elif node.op == "output":
            # If this is the last node, we should add an output node and add the last graph to the list.
            assert prev_node, "prev_node cannot be None"
            with new_graph.inserting_after(prev_node):
                new_node = new_graph.node_copy(node, lambda x: env[x.name])
            new_graph.lint()
            module_list.append(torch.fx.GraphModule(model, new_graph))
            break

        prev_node = new_node
        prev_shard_id = node_name_to_shard_id[node.name]

    return module_list

model_configs = {
    "gpt3_1_3B": {"microbatch": 4, "world_sizes": list(range(9, 17))},
    "gpt3_2_7B": {"microbatch": 4, "world_sizes": list(range(10, 17))},
    # "gpt3_350M": {"microbatch": 8, "world_sizes": list(range(8, 16))},
}
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

def get_model_layers(model_tag):
        if model_tag == "gpt3_2_7B":
            model_config =  AutoConfig.from_pretrained("/workspace/Oobleck/data/model/gpt3_2_7B/config.json")
        elif model_tag == "gpt3_1_3B":
            model_config = AutoConfig.from_pretrained("/workspace/Oobleck/data/model/gpt3_1_3B/config.json")
        elif model_tag == "gpt3_6_7B":
            model_config = AutoConfig.from_pretrained("/workspace/Oobleck/data/model/gpt3_6_7B/config.json")
        elif model_tag == "gpt3_350M":
            model_config = AutoConfig.from_pretrained("/workspace/Oobleck/data/model/gpt3_350M/config.json")
        else:
            raise Exception(f"No model config for model: {model_tag}")
        model = AutoModelForPreTraining.from_config(model_config, torch_dtype=torch.float16)
        sample_inputs = 
        trace_input_names = list(sample_inputs.keys())
        split_points = get_split_points(model_config)
        layers = shard_model(model, trace_input_names, split_points)
        return layers

def get_stage_parameters(file_path: str, model_tag):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            # 读取并解析JSON文件
            data = json.load(file)
            pipelines = data["pipelines"]
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    layers = get_model_layers(model_tag)
    for pipeline in pipelines:
        num_layers_per_stage = pipeline["num_layers_per_stage"]
        for num_layers in num_layers_per_stage:
            end_idx += num_layers
            get_parameters(layers[start_idx: start_idx + num_layers])
            start_idx = 0

def get_parameters(layers):
    num_params = sum(
            sum(p.numel() for p in layer.parameters()) for layer in layers
        )
    print(f"params: {num_params}")
    return num_params


if __name__ == "__main__":
    for model, config in model_configs.items():
        microbatch = config["microbatch"]
        for world_size in config["world_sizes"]:
            label = f"{model}-{microbatch}-{world_size}"
            layer_file = f"/workspace/Oobleck/important_data/pipelines/{label}.json"
            get_stage_parameters(layer_file, model)