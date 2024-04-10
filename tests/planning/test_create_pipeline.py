from oobleck.csrc.planning.pipeline_template import (
    LayerExecutionResults,
    PipelineTemplateGenerator,
    StageExecutionResult,
    get_profile_results,
    PipelineTemplate
)

# min_nodes: 2, max_nodes: 4, gpus: 1
min_nodes = 2
max_nodes = 4
num_gpus_per_node = 1

model_name = "gpt2"
model_tag = "medium"
batch_size = 8
profile_results: LayerExecutionResults = get_profile_results(model_name, model_tag, batch_size)

print(f"profile results: size {profile_results.size}")
layer_exe_results = profile_results.get()
for layer_result in layer_exe_results:
    print(f"layer idx: {layer_result._index}, allreduce_across_nodes: {layer_result._allreduce_across_nodes}, allreduce_in_node: {layer_result._allreduce_in_node}, forward: {layer_result._forward}, backward: {layer_result._backward}, mem: {layer_result._mem_required}")

geneator = PipelineTemplateGenerator()


pipeline_templates = geneator.create_pipeline_templates_serial(profile_results, (min_nodes, max_nodes), num_gpus_per_node)

print("pipeline templates")
pipeline_id = 0
for pipeline in pipeline_templates:
    print(f"pl {pipeline_id}: rank_grid: {pipeline.get_rank_grid(list(range(max_nodes)))}, num_nodes: {pipeline._num_nodes}, num_gpu_per_node: {pipeline._num_gpus_per_node}, iter: {pipeline._iteration_time}")
    stage_id = 0
    for stage in pipeline.get_stages():
        print(f"stage {stage_id}: layer_indices {stage._layer_indices}, mem: {stage._mem_required}, num_gpus: {stage._num_gpus}")
        stage_id += 1
    pipeline_id += 1