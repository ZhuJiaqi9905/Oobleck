import pytest

from oobleck.csrc.planning.pipeline_template import (
    LayerExecutionResults,
    PipelineTemplateGenerator,
)
from tests.conftest import OobleckSingleProcessTestCase


class TestOobleckPipelineTemplate(OobleckSingleProcessTestCase):
    @pytest.fixture(scope="function")
    def profile(self) -> LayerExecutionResults:
        return self.factory.get_dummy_profile()

    def test_create_pipeline_templates_onegpu(self, profile: LayerExecutionResults):
        generator = PipelineTemplateGenerator()
        pipeline_templates = generator.create_pipeline_templates(
            profile,
            (1, 1),  # num nodes range
            1,
        )
        print(pipeline_templates)
        assert len(pipeline_templates) == 1
        assert pipeline_templates[0]._num_nodes == 1
        assert pipeline_templates[0]._num_gpus_per_node == 1
        assert len(pipeline_templates[0].get_stages()) == 1
        assert pipeline_templates[0]._iteration_time > 0

    def test_create_pipeline_templates_maxnode(self, profile: LayerExecutionResults):
        generator = PipelineTemplateGenerator()
        num_nodes = profile.size  # num_nodes == number of layers
        pipeline_templates = generator.create_pipeline_templates(
            profile,
            (num_nodes, num_nodes),  # num nodes range
            1,
        )
        assert len(pipeline_templates) == 1
        assert pipeline_templates[0]._num_nodes == num_nodes
        assert pipeline_templates[0]._num_gpus_per_node == 1
        assert len(pipeline_templates[0].get_stages()) == num_nodes
        assert pipeline_templates[0]._iteration_time > 0

    def test_create_pipeline_templates_too_many_nodes(
        self, profile: LayerExecutionResults
    ):
        generator = PipelineTemplateGenerator()
        num_nodes = profile.size + 1
        pipeline_templates = generator.create_pipeline_templates(
            profile,
            (num_nodes, num_nodes),  # num nodes range
            1,
        )
        assert len(pipeline_templates) == 0

    def test_create_pipeline_templates_node_range(self, profile: LayerExecutionResults):
        generator = PipelineTemplateGenerator()
        max_num_nodes = profile.size
        pipeline_templates = generator.create_pipeline_templates(
            profile,
            (2, 8),  # num nodes range
            1,
        )
        assert 0 < len(pipeline_templates) <= max_num_nodes
        assert 0 < pipeline_templates[0]._num_nodes <= max_num_nodes
        for pipeline_template in pipeline_templates:
            assert pipeline_templates[0]._num_gpus_per_node == 1
            assert 2 <= len(pipeline_template.get_stages()) <= 8
            assert pipeline_template._iteration_time > 0

    def test_create_pipeline_templates_multiple_gpus_in_node(
        self, profile: LayerExecutionResults
    ):
        generator = PipelineTemplateGenerator()
        pipeline_templates = generator.create_pipeline_templates(profile, (1, 1), 4)
        assert len(pipeline_templates) >= 1
        sum(
            template._num_gpus_per_node * template._num_nodes
            for template in pipeline_templates
        ) == 4

    def test_create_pipeline_templates_multiple_gpus_in_node_range(
        self, profile: LayerExecutionResults
    ):
        generator = PipelineTemplateGenerator()
        pipeline_templates = generator.create_pipeline_templates(profile, (1, 6), 4)
        assert len(pipeline_templates) >= 1
        for index, template in enumerate(pipeline_templates):
            num_nodes = index + 1
            assert template._num_gpus_per_node == 4
            assert num_nodes == template._num_nodes

    @pytest.mark.skip(reason="Not implemented yet")
    def test_create_pipeline_templates_fsdp(self, profile: LayerExecutionResults):
        pass
