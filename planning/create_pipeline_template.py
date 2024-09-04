from oobleck.csrc.planning.pipeline_template import (
    LayerExecutionResults,
    PipelineTemplate,
    PipelineTemplateGenerator,
    StageExecutionResult,
    get_profile_results,
)
from deepspeed.utils.logging import LoggerFactory, log_dist
from deepspeed.utils.logging import logging
import math
import argparse
import torch
import os
from oobleck.planning.instantiator import (
    HeterogeneousPipelinesExecutionPlan,
    PipelineInstantiator,
)
import json

logger = LoggerFactory.create_logger("oobleck_engine", logging.DEBUG)


def pipeline_template_to_pipeline(template: PipelineTemplate):
    pipeline_stages = []
    for stage in template.get_stages():
        pipeline_stages.append(
            {"num_gpus": stage._num_gpus, "layer_indices": stage._layer_indices}
        )

    return {
        "num_nodes": template._num_nodes,
        "num_gpus_per_node": template._num_gpus_per_node,
        "stages": pipeline_stages,
        "iteration_time": template._iteration_time,
    }


def generate_pipeline_templates(
    model: str,
    micro_batch_size: int,
    num_nodes: int,
    num_gpus_per_node: int,
    cuda_memory: int | None,
    global_batch_size: int,
):

    profile_results: LayerExecutionResults = get_profile_results(
        model, micro_batch_size, num_gpus_per_node * num_nodes, num_gpus_per_node
    )

    total_memory_consumption = 6 * sum(
        [layer_result._mem_required[0] for layer_result in profile_results.get()]
    )
    total_memory_consumption += max(
        [layer_result._mem_required[1] for layer_result in profile_results.get()]
    )
    if cuda_memory == None:
        cuda_memory = torch.cuda.get_device_properties("cuda:0").total_memory
    min_num_nodes = max(
        1,
        math.ceil(total_memory_consumption / (cuda_memory * num_gpus_per_node)),
    )
    max_num_nodes = num_nodes
    assert min_num_nodes <= max_num_nodes, (
        "Minimum required number of nodes is larger than maximum number of nodes "
        f"(minimum required: {min_num_nodes}, you have: {max_num_nodes})."
    )

    logger.info(f"Number of nodes range: ({min_num_nodes}, {max_num_nodes})")

    # TODO: Calculate num_gpus_range based on profile results
    print(
        f"min_nodes: {min_num_nodes}, max_nodes: {max_num_nodes}, gpus: {num_gpus_per_node}, "
    )
    template_generator = PipelineTemplateGenerator()

    pipeline_templates: list[PipelineTemplate] = (
        template_generator.create_pipeline_templates(
            profile_results, (min_num_nodes, max_num_nodes), num_gpus_per_node
        )
    )
    pipelines = []
    for pipeline_template in pipeline_templates:
        pipelines.append(pipeline_template_to_pipeline(pipeline_template))

    # generate microbatch assignment
    instantiator = PipelineInstantiator()
    global_num_microbatch = global_batch_size // micro_batch_size
    execution_plan: HeterogeneousPipelinesExecutionPlan = (
        instantiator.get_best_execution_plan(
            pipeline_templates,
            [layer._allreduce_across_nodes for layer in profile_results.get()],
            num_nodes,
            global_num_microbatch,
        )
    )

    json_execution_plan = {}
    json_execution_plan["pipeline_templates"] = [
        pipeline_template_to_pipeline(pipeline)
        for pipeline in execution_plan.pipeline_templates
    ]
    json_execution_plan["num_instances_set"] = [
        (pipeline_template_to_pipeline(pipeline), val)
        for pipeline, val in execution_plan.num_instances_set.items()
    ]
    json_execution_plan["num_microbatches_set"] = [
        (pipeline_template_to_pipeline(pipeline), val)
        for pipeline, val in execution_plan.num_microbatches_set.items()
    ]
    json_execution_plan["allreduce_across_nodes"] = (
        execution_plan.allreduce_across_nodes
    )

    return {"pipeline_template": pipelines, "execution_plan": json_execution_plan}


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
        if len(pipeline_template.get_stages()) != len(json_pipeline["stages"]) or pipeline_template._num_nodes != json_pipeline["num_nodes"] or pipeline_template._num_gpus_per_node != json_pipeline["num_gpus_per_node"]:
            continue
        is_same = True
        for i in range(len(json_pipeline["stages"])):
            stage = pipeline_template.get_stages()[i]
            json_stage = json_pipeline["stages"][i]
            layer_indices = stage._layer_indices
            json_layer_indices = json_stage["layer_indices"]
            if stage._num_gpus != json_stage["num_gpus"] or len(layer_indices) != len(json_layer_indices): 
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


def test():
    file_path = "/workspace/Oobleck/tmp/pipeline_templates/gpt3_1_3B-16-12-1.json"
    pipeline_templates = create_pipeline_templates_from_json_file(
        file_path
    )
    plan = create_heterogeneous_plan_from_json_file(file_path, pipeline_templates) 

    print(f"plan: {plan}")
    print("pipeline_templates")
    for template in plan.pipeline_templates:
        print(template)
    print("num_instances_set")
    for template, val in plan.num_instances_set.items():
        print(f"{template}: {val}")
    print("num_microbatches_set")
    for template, val in plan.num_microbatches_set.items():
        print(f"{template}: {val}")
    

    
if __name__ == "__main__":
    test()
    exit()
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-path", type=str)
    parser.add_argument("--out-path", type=str)
    parser.add_argument("--cuda-memory", type=int)
    # global batch size
    parser.add_argument("--gbs", type=int)
    args = parser.parse_args()

    for file_name in os.listdir(args.profile_path):

        # hack
        file_name = "gpt3_1_3B-16-12-1"
        print(f"{file_name}")

        parts = file_name.split("-")
        model = parts[0]
        micro_batch_size = int(parts[1])
        num_nodes = int(parts[2])
        num_gpus_per_node = int(parts[3])

        result = generate_pipeline_templates(
            model,
            micro_batch_size,
            num_nodes,
            num_gpus_per_node,
            args.cuda_memory,
            args.gbs,
        )

        out_file = os.path.join(args.out_path, f"{file_name}.json")
        with open(out_file, "w") as fp:
            json.dump(result, fp)
        print(f"result: {result}")

        break
