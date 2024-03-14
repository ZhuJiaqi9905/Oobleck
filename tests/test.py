import math
num_gpus_per_node = 4
num_gpus_list = [2**i for i in range(int(math.log2(num_gpus_per_node)) + 1)]
ranks = list(range(num_gpus_per_node))

process_groups = []
for i in range(len(num_gpus_list)):
    pg_ranks = ranks[: num_gpus_list[i]]
    print(f"pg_ranks is {pg_ranks}" )
    # process_groups.append(
    #     (dist.get_rank() in pg_ranks, dist.new_group(pg_ranks))
    # )

# results: list[list[int]] = [
#     [0] * len(process_groups) for _ in range(len(self.model.layers))
# ]