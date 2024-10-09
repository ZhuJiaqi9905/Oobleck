import subprocess
import itertools
import threading
import time
import os




MODEL_CONFIGS = {
    "gpt3_1_3B": {
        "microbatch": 16,
        "world_sizes": {
            9: [8],
            10: [8, 9],
            11: [10],
            12: [10, 11],
            13: [11, 12],
            14: [12, 13],
            15: [14],
            16: [14, 15],
            17: [14, 15, 16],
            18: [12, 14, 15, 16, 17],
            19: [16, 17, 18],
            20: [16, 17, 18, 19],
        },
    },
    "gpt3_2_7B": {
        "microbatch": 8,
        "world_sizes": {
            10: [8, 9],
            11: [10],
            12: [10, 11],
            13: [11, 12],
            14: [12, 13],
            15: [14],
            16: [14, 15],
            17: [14, 15, 16],
            18: [12, 14, 15, 16, 17],
            19: [16, 17, 18],
            20: [16, 17, 18, 19],
        },
    },
    # "gpt3_6_7B": {
    #     "microbatch": 2,
    #     "world_sizes": {
    #         10: [8, 9],
    #         11: [10],
    #         12: [10, 11],
    #         13: [11, 12],
    #         14: [12, 13],
    #         15: [14],
    #         16: [14, 15],
    #         17: [14, 15, 16],
    #         18: [12, 14, 15, 16, 17],
    #         19: [16, 17, 18],
    #         20: [16, 17, 18, 19],
    #     },
    # },
    # "gpt3_350M": {
    #     "microbatch": 32,
    #     "world_sizes": {
    #         10: [8, 9],
    #         11: [10],
    #         12: [10, 11],
    #         13: [11, 12],
    #         14: [12, 13],
    #         15: [14],
    #         16: [14, 15],
    #         17: [14, 15, 16],
    #         18: [12, 14, 15, 16, 17],
    #         19: [16, 17, 18],
    #         20: [16, 17, 18, 19],
    #     },
    # },
}
# MODEL_CONFIGS = {
#     "gpt3_1_3B": {
#         "microbatch": 16,
#         "world_sizes": {
#             15: [13],
#             16: [13],
#             20: [15],
#         },
#     },
#     "gpt3_2_7B": {
#         "microbatch": 8,
#         "world_sizes": {
#             15: [13],
#             16: [13],
#             20: [15],
#         },
#     },
#     "gpt3_6_7B": {
#         "microbatch": 2,
#         "world_sizes": {
#             15: [13],
#             16: [13],
#             20: [15],
#         },
#     }
# }

MODEL_CONFIGS = {
    "gpt3_2_7B": {
        "microbatch": 8,
        "world_sizes": {
            13: [12]
        },
    },    
}
# 超时时间（秒）
TIMEOUT_SECONDS = 600
CUDA_MEMORY = 16945709056
OUT_DIR = "/workspace/Oobleck/important_data/nsdi/lost_nodes/oobleck_tmp/"
COMMAND_TEMPLATE = "python /workspace/Oobleck/simulate/oobleck_simulate.py --model {} --microbatch {} --worldsize {}  --lost-nodes {} --out-dir {} --cuda-memory {}"


def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command '{command}' failed with error: {e}")


def run_with_timeout(command, timeout):
    proc = subprocess.Popen(command, shell=True)
    timer = threading.Timer(timeout, proc.kill)
    try:
        timer.start()
        proc.communicate()
    finally:
        timer.cancel()


# 生成所有参数组合
for model, config in MODEL_CONFIGS.items():
    microbatch = config["microbatch"]
    nodes_change = config["world_sizes"]
    for ori_world_size, curr_world_sizes in nodes_change.items():
        for curr_world_size in curr_world_sizes:
            if not os.path.exists(
                f"/workspace/Oobleck/planning/pipeline_templates/{model}-{microbatch}-{ori_world_size}-1.json"
            ):
                continue 
            command = COMMAND_TEMPLATE.format(
                model, microbatch, ori_world_size, ori_world_size - curr_world_size, OUT_DIR, CUDA_MEMORY
            )
            print(f"Running command: {command}")
            run_with_timeout(command, TIMEOUT_SECONDS)