
from oobleck.csrc.planning.pipeline_template import (
    LayerExecutionResults,
    PipelineTemplateGenerator,
    StageExecutionResult,
    get_profile_results,
    PipelineTemplate
)

geneator = PipelineTemplateGenerator()
num_nodes = 4
model_name = "gpt2"
model_tag = "medium"
microbatch_size = 8
profile: LayerExecutionResults = get_profile_results(
    model_name,
    model_tag,
    microbatch_size
)
pipeline_templates = geneator.create_pipeline_templates(profile, (1, num_nodes), 1)
# l = []
# l.append(s)
# s = StageExecutionResult(, [], 4)
# p = PipelineTemplate(l, 100, 40, 4, 1)
# print(p)
for pipeline in pipeline_templates:
    print(pipeline)
    print(pipeline.get_rank_grid([0, 1, 2, 3]))
    stages = pipeline.get_stages()
    for stage in stages:
        print(stage._layer_indices)
        print(stage._mem_required)
        print(stage._num_gpus)
    
    