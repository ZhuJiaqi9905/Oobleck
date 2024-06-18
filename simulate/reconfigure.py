# 每条pipeline。对于每个pipeline，是layer -> [gpu ranks]


class Pipeline:
    def __init__(self, layers: dict[int, list[int]]) -> None:
        self._ranks: list[int] = []
        for layer, ranks in layers.items():
            self._ranks.extend(ranks)
        self._ranks = list(set(self._ranks)).sort()


rank_grids: list[dict[int, list[int]]] = [
    {
        0: [0],
        1: [0],
        2: [0],
        3: [0],
        4: [0],
        5: [0],
        6: [0],
        7: [0],
        8: [0],
        9: [0],
        10: [0],
        11: [0],
        12: [0],
        13: [0],
        14: [0],
        15: [0],
        16: [0],
        17: [0],
        18: [0],
        19: [0],
        20: [0],
        21: [0],
        22: [0],
        23: [0],
        24: [0],
        25: [0],
        26: [0],
        27: [0],
        28: [0],
        29: [0],
        30: [0],
        31: [0],
        32: [1],
        33: [1],
        34: [1],
        35: [1],
        36: [1],
        37: [1],
        38: [1],
        39: [1],
        40: [1],
        41: [1],
        42: [1],
        43: [1],
        44: [1],
        45: [1],
        46: [1],
        47: [1],
        48: [1],
        49: [1],
        50: [1],
        51: [1],
        52: [1],
        53: [1],
        54: [1],
        55: [1],
        56: [1],
        57: [1],
        58: [1],
        59: [1],
        60: [1],
        61: [1],
    },
    {
        0: [2],
        1: [2],
        2: [2],
        3: [2],
        4: [2],
        5: [2],
        6: [2],
        7: [2],
        8: [2],
        9: [2],
        10: [2],
        11: [2],
        12: [2],
        13: [2],
        14: [2],
        15: [2],
        16: [2],
        17: [2],
        18: [2],
        19: [2],
        20: [2],
        21: [2],
        22: [3],
        23: [3],
        24: [3],
        25: [3],
        26: [3],
        27: [3],
        28: [3],
        29: [3],
        30: [3],
        31: [3],
        32: [3],
        33: [3],
        34: [3],
        35: [3],
        36: [3],
        37: [3],
        38: [3],
        39: [3],
        40: [3],
        41: [3],
        42: [4],
        43: [4],
        44: [4],
        45: [4],
        46: [4],
        47: [4],
        48: [4],
        49: [4],
        50: [4],
        51: [4],
        52: [4],
        53: [4],
        54: [4],
        55: [4],
        56: [4],
        57: [4],
        58: [4],
        59: [4],
        60: [4],
        61: [4],
    },
]

lost_ranks: list[int] = [2, 3]


def reconfigure(
    rank_grids: list[dict[int, list[int]]], lost_ranks: list[int], min_num_ranks: int
):
    pipelines: list[Pipeline] = []

    for layers in rank_grids:
        pipelines.append(Pipeline(layers))

    # Update ranks first
    for pipeline in pipelines:
        pipeline._ranks = [rank for rank in pipeline._ranks if rank not in lost_ranks]
    need_merge: bool = False
    new_ranks_list: list[list[int]] = []
    # Prepare new instances set.
    for pipeline in pipelines:
        ranks = pipeline._ranks

        # If all ranks are gone, skip it.
        if len(ranks) == 0:
            continue

        # If there is an available template, use it.
        if len(ranks) >= min_num_ranks:
            new_ranks_list.append(ranks)
            continue

        # This pipeline needs more ranks
        biggest_pipeline: Pipeline = None
        while len(ranks) < min_num_ranks:
            biggest_pipeline = find_biggest_pipeline(pipelines, min_num_ranks)
            if biggest_pipeline is None:
                # No pipelines can yield a rank. Simply add it and handle later.
                need_merge = True
                break

            while (
                len(biggest_pipeline._ranks) > min_num_ranks
                and len(ranks) < min_num_ranks
            ):
                ranks.append(biggest_pipeline._ranks.pop())

        new_ranks_list.append(ranks)
    print(f"need_merge: {need_merge}, new_ranks_list: {new_ranks_list}")
    # Merge pipelines if needed new_ranks_list: [[0], [1], [2]]
    if need_merge:
        new_ranks_list = merge_pipelines(new_ranks_list)

    # sort ranks for each list of ranks
    for ranks in new_ranks_list:
        ranks.sort()

    # Sort ranks by length so that smaller pipeline ranks always come first.
    # For pipelines with the same number of ranks, a pipeline with smaller rank id comes first.
    new_ranks_list.sort(key=lambda ranks: (len(ranks), ranks[0]))

    print(f"new_ranks_list: {new_ranks_list}")

    # Create new instances set
    new_num_instances_set: dict[PipelineTemplate, int] = defaultdict(int)
    for ranks in new_ranks_list:
        template = get_pipeline_template(ranks, self.engine._pipeline_templates)
        if template != None:
            new_num_instances_set[template] += 1

    # new_pipeline = self._reinstantiate(new_num_instances_set, new_ranks_list)

    # Copy model states here
    new_rank_grids: list[dict[int, list[int]]] = []
    for pipeline_template, num_instance in new_num_instances_set.items():
        for _ in range(num_instance):
            rank_grid = pipeline_template.get_rank_grid(new_ranks_list.pop(0))
            new_rank_grids.append(rank_grid)


def get_pipeline_template(
    ranks: list[int], pipeline_templates: list[PipelineTemplate]
) -> PipelineTemplate | None:
    return next(
        (
            template
            for template in pipeline_templates
            if template._num_nodes * template._num_gpus_per_node == len(ranks)
        ),
        None,
    )


def find_biggest_pipeline(pipelines: list[Pipeline], min_num_ranks: int) -> Pipeline | None:
    biggest_pipeline: Pipeline | None = None
    for pipeline in pipelines:
        if biggest_pipeline is None or len(pipeline._ranks) >= len(
            biggest_pipeline._ranks
        ):
            biggest_pipeline = pipeline

    # Check if this pipeline can yield a node
    if biggest_pipeline and len(biggest_pipeline._ranks) > min_num_ranks:
        return biggest_pipeline

    return None

def merge_pipelines(ranks_list: list[list[int]], min_num_ranks: int) -> list[list[int]]:
    """
    When this method is called, all pipelines cannot yield a rank
    but still there is a pipeline that needs more ranks.
    Solve this problem by merging at least two pipelines.

    Return: list of ranks for a merged pipeline and remaining pipelines.
    """
    ranks_to_merge: list[list[int]] = []
    results: list[list[int]] = []
    for ranks in ranks_list:
        ranks_to_merge.append(ranks) if len(
            ranks
        ) < min_num_ranks else results.append(ranks)

    try:
        # Merge pipelines
        while ranks_to_merge:
            ranks = ranks_to_merge.pop(0)
            try:
                while len(ranks) < min_num_ranks:
                    ranks.extend(ranks_to_merge.pop(0))
            except IndexError:
                # No more ranks to merge.
                # Get ranks from result pipeline
                ranks.extend(results.pop(0))

            assert len(ranks) >= min_num_ranks
            results.append(ranks)
    except IndexError:
        raise RuntimeError("Ranks are insufficient")

    assert ranks_to_merge == []
    return results