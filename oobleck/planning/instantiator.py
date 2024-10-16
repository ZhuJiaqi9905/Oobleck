from collections import defaultdict
from typing import Optional

import pyomo.environ as pyomo
from deepspeed import comm as dist
from deepspeed.utils import logger
from transformers.training_args import TrainingArguments

from oobleck.csrc.planning.pipeline_template import PipelineTemplate
from oobleck.execution.dataloader import OobleckDataLoader
from oobleck.execution.pipeline import OobleckPipeline
from oobleck.module.model import OobleckModel


class HeterogeneousPipelinesExecutionPlan:
    def __init__(
        self,
        pipeline_templates: list[PipelineTemplate],
        num_instances_set: dict[PipelineTemplate, int],
        num_microbatches_set: dict[PipelineTemplate, int],
        allreduce_across_nodes: list[dict[int, float]],
    ):
        pipeline_templates = [
            template
            for template in pipeline_templates
            if template in num_instances_set and template in num_microbatches_set
        ]
        pipeline_templates.sort(key=lambda template: template._num_nodes)

        self.pipeline_templates = pipeline_templates
        self.num_instances_set = num_instances_set
        self.num_microbatches_set = num_microbatches_set
        self.allreduce_across_nodes = allreduce_across_nodes

        self.total_num_pipelines = sum(
            num_instances for num_instances in num_instances_set.values()
        )
        self.total_num_microbatches = sum(
            self.num_instances_set[template] * self.num_microbatches_set[template]
            for template in self.pipeline_templates
        )

    def __repr__(self) -> str:
        result = ""
        for template in self.pipeline_templates:
            result += (
                f"{self.num_instances_set[template]} x {template} pipelines "
                f"(b: {self.num_microbatches_set[template]})\n"
            )
        result += f"B: {self.total_num_microbatches}"
        return result

    @property
    def iteration_time(self) -> float:
        # TODO: should be divided by number of stages?
        max_iteration_time = max(
            pipeline_template._iteration_time * num_microbatches
            for pipeline_template, num_microbatches in self.num_microbatches_set.items()
        )

        # FIXME: currently we only consider communication overhead
        # of the first layer, believing communication of other layers
        # can fully be hidden in backward pass computing.
        synchronization_overhead = self.allreduce_across_nodes[0][
            self.total_num_pipelines
        ]

        return max_iteration_time + synchronization_overhead

    @property
    def my_pipeline_index(self) -> int:
        my_rank = dist.get_rank()

        start_rank = 0
        pipeline_index = 0
        for pipeline_template, num_instances in self.num_instances_set.items():
            for _ in range(num_instances):
                num_gpus_per_template = (
                    pipeline_template._num_nodes * pipeline_template._num_gpus_per_node
                )

                if (
                    my_rank >= start_rank
                    and my_rank < start_rank + num_gpus_per_template
                ):
                    return pipeline_index

                start_rank += num_gpus_per_template
                pipeline_index += 1

        raise RuntimeError("This rank is not in any pipeline")

    @property
    def num_microbatches(self) -> list[int]:
        results: list[int] = []
        for pipeline_template in self.pipeline_templates:
            results.extend(
                [self.num_microbatches_set[pipeline_template]]
                * self.num_instances_set[pipeline_template]
            )
        return results

    def instantiate(
        self,
        model: OobleckModel,
        dataloader: OobleckDataLoader,
        training_args: TrainingArguments,
        num_gpus_per_node: int,
        ranks: list[list[int]] | None = None,
        step: int = 0,
        # layer_index -> dict of [fsdp_index -> ProcessGroup]
    ) -> tuple[OobleckPipeline, list[OobleckPipeline]]:
        my_pipeline: Optional[OobleckPipeline] = None
        pipeline_index: int = 0
        num_ranks_used = 0

        all_pipelines: list[OobleckPipeline] = []
        for pipeline_template, num_instances in self.num_instances_set.items():
            for _ in range(num_instances):
                pipeline_ranks: list[int] = (
                    list(
                        range(
                            num_ranks_used,
                            num_ranks_used
                            + pipeline_template._num_nodes * num_gpus_per_node,
                        )
                    )
                    if ranks is None
                    else ranks[pipeline_index]
                )
                assert pipeline_template._num_nodes * num_gpus_per_node == len(
                    pipeline_ranks
                )

                pipeline = OobleckPipeline(
                    pipeline_id=pipeline_index,
                    pipeline_template=pipeline_template,
                    ranks=pipeline_ranks,
                    dataloader=dataloader,
                    step=step,
                    training_args=training_args,
                )

                all_pipelines.append(pipeline)
                if pipeline.my_pipeline:
                    my_pipeline = pipeline

                pipeline_index += 1
                num_ranks_used += len(pipeline_ranks)
        
        assert my_pipeline is not None
        return my_pipeline, all_pipelines
    
    def instantiate_simulated(
        self,
        model: OobleckModel,
        dataloader: OobleckDataLoader,
        training_args: TrainingArguments,
        num_gpus_per_node: int,
        ranks: list[list[int]] | None = None,
        step: int = 0,
        # layer_index -> dict of [fsdp_index -> ProcessGroup]
    ) -> tuple[OobleckPipeline, list[OobleckPipeline]]:
        my_pipeline: Optional[OobleckPipeline] = None
        pipeline_index: int = 0
        num_ranks_used = 0

        all_pipelines: list[OobleckPipeline] = []
        for pipeline_template, num_instances in self.num_instances_set.items():
            for _ in range(num_instances):
                pipeline_ranks: list[int] = (
                    list(
                        range(
                            num_ranks_used,
                            num_ranks_used
                            + pipeline_template._num_nodes * num_gpus_per_node,
                        )
                    )
                    if ranks is None
                    else ranks[pipeline_index]
                )
                assert pipeline_template._num_nodes * num_gpus_per_node == len(
                    pipeline_ranks
                )

                pipeline = OobleckPipeline(
                    pipeline_id=pipeline_index,
                    pipeline_template=pipeline_template,
                    ranks=pipeline_ranks,
                    dataloader=dataloader,
                    step=step,
                    training_args=training_args,
                    is_simulated=True
                )

                all_pipelines.append(pipeline)
                if pipeline.my_pipeline:
                    my_pipeline = pipeline

                pipeline_index += 1
                num_ranks_used += len(pipeline_ranks)
        
        assert my_pipeline is not None
        return my_pipeline, all_pipelines

class PipelineInstantiator:
    def get_best_execution_plan(
        self,
        pipeline_templates: list[PipelineTemplate],
        allreduce_across_nodes: list[dict[int, float]],
        num_nodes: int,
        global_num_microbatch: int,
    ) -> HeterogeneousPipelinesExecutionPlan:
        """
        Section 4.2. Pipeline Instantiation implementation
        """
        num_instances_set_list: list[
            dict[PipelineTemplate, int]
        ] = self._enumerate_instantiation_options(pipeline_templates, num_nodes)

        num_microbatches_set_list: list[dict[PipelineTemplate, int]] = []
        for num_instances_set in num_instances_set_list:
            try:
                num_microbatches_set = self._distribute_batch(
                    global_num_microbatch, num_instances_set
                )
                num_microbatches_set_list.append(num_microbatches_set)
            except ValueError:
                num_microbatches_set_list.append(None)

        execution_plans: list[HeterogeneousPipelinesExecutionPlan] = [
            HeterogeneousPipelinesExecutionPlan(
                pipeline_templates,
                num_instances_set,
                num_microbatches_set,
                allreduce_across_nodes,
            )
            for num_instances_set, num_microbatches_set in zip(
                num_instances_set_list, num_microbatches_set_list
            )
            if num_instances_set is not None and num_microbatches_set is not None
        ]

        assert execution_plans, "No feasible execution plan found."

        result: HeterogeneousPipelinesExecutionPlan = min(
            execution_plans, key=lambda plan: plan.iteration_time
        )
        
        logger.info(f"Best execution plan: {result}")
        return result

    def get_new_execution_plan(
        self,
        new_num_instances_set: dict[PipelineTemplate, int],
        allreduce_across_nodes: list[dict[int, float]],
        global_num_microbatch: int,
    ) -> HeterogeneousPipelinesExecutionPlan:
        # it is None
        num_microbatches_set: dict[PipelineTemplate, int] = self._distribute_batch(
            global_num_microbatch, new_num_instances_set
        )

        execution_plan: HeterogeneousPipelinesExecutionPlan = (
            HeterogeneousPipelinesExecutionPlan(
                list(new_num_instances_set.keys()),
                new_num_instances_set,
                num_microbatches_set,
                allreduce_across_nodes,
            )
        )

        logger.info(f"New execution plan: {execution_plan}")
        return execution_plan

    def _enumerate_instantiation_options(
        self,
        pipeline_templates: list[PipelineTemplate],
        num_nodes: int,
    ) -> list[dict[PipelineTemplate, int]]:
        """Oobleck paper section 4.2.1. Enumerating instantiation options implementation
        Get all feasible sets of xi's that can use all of the given nodes.
        """

        dp: list[list[list[dict[PipelineTemplate, int]]]] = [
            [[] for _ in range(num_nodes + 1)]
            for _ in range(len(pipeline_templates) + 1)
        ]

        for i in range(1, len(pipeline_templates) + 1):
            dp[i][0] = [defaultdict(int)]
            for j in range(1, num_nodes + 1):
                # (1) in Figure: copy all dicts
                dp[i][j] = [combo.copy() for combo in dp[i - 1][j]]
                if pipeline_templates[i - 1]._num_nodes <= j:
                    # (2) in Figure: copy all dicts with one pipeline_templates[i - 1] added
                    for combo in dp[i][j - pipeline_templates[i - 1]._num_nodes]:
                        new_combo = combo.copy()
                        new_combo[pipeline_templates[i - 1]] += 1

                        # concatenate two lists
                        dp[i][j].append(new_combo)

        return dp[-1][-1]

    def _distribute_batch(
        self,
        global_num_microbatch: int,
        num_instances_set: dict[PipelineTemplate, int],
    ) -> Optional[dict[PipelineTemplate, int]]:
        """Oobleck paper section 4.2.2. Batch distribution implementation
        satisfying the following two requirements
        1. std(Bi * Ti) is minimized
        2. sum(Bi) = B
        3. Each Bi is divisible by b (training_args.per_device_train_batch_size)

        Bi是第i个pipeline的accumulated micro batch size

        Use Pyomo (Python Optimization Modeling Objects).

        Args:
            global_num_microbatch (int): Global batch // training_args.per_device_train_batch_size.
            instances_per_pp_template (list[int]): list of number of pipeline instances per template
            pipeline_iteration_time (list[float]): list of PipelineSpec.e.

        Returns:
            list[int]: list of number of microbatch per template.
                Multiply by training_args.per_device_train_batch_size to calculate minibatch size.
        """

        model = pyomo.ConcreteModel()

        model.I = pyomo.Set(initialize=list(range(len(num_instances_set))))

        T = {
            i: pipeline_template._iteration_time
            for i, pipeline_template in enumerate(num_instances_set)
        }

        x = {
            i: instance_num for i, instance_num in enumerate(num_instances_set.values())
        }
        s = {
            i: len(pipeline_template.get_stages())
            for i, pipeline_template in enumerate(num_instances_set.keys())
        }

        # Define the Pyomo variable
        # nb: number fo microbatches per PipelineSpec
        model.nb = pyomo.Var(model.I, within=pyomo.PositiveIntegers)

        # Objective function
        def objective(model):
            avg_bT = sum(T[i] / s[i] * model.nb[i] for i in model.I) / len(model.I)
            return sum((T[i] / s[i] * model.nb[i] - avg_bT) ** 2 for i in model.I)

        model.obj = pyomo.Objective(rule=objective, sense=pyomo.minimize)

        # Define constraints
        def c1(model):
            return sum(model.nb[i] * x[i] for i in model.I) == global_num_microbatch

        # def c2(model, i):
        #     return model.nb[i] >= 2 * s[i]

        model.constraint1 = pyomo.Constraint(rule=c1)
        # model.constraint2 = pyomo.Constraint(range(len(model.I)), rule=c2)

        pyomo.SolverFactory("mindtpy").solve(
            model, mip_solver="glpk", nlp_solver="ipopt", tee=False
        )

        # check for all i model.nb[i].value is integer, otherwise return None
        # 这里返回了None
        if not all(model.nb[i].value for i in model.I):
            logger.info("Batch distribution find no results. return None")
            return None

        nb_optimal = {
            spec: int(model.nb[i].value)
            for i, spec in zip(model.I, num_instances_set.keys())
        }

        logger.debug(f"Number of microbatch per PipelineSpec: {nb_optimal}")
        return nb_optimal
