import subprocess
import itertools
import threading
import time
import os


MODEL_CONFIGS = {
    "gpt3_1_3B": {"microbatch": 16, "world_sizes": list(range(8, 21, 1)), "lost_nodes": list(range(1, 2))},
    "gpt3_2_7B": {"microbatch": 8, "world_sizes": list(range(8, 21, 1)), "lost_nodes": list(range(1, 2))},
    "gpt3_6_7B": {"microbatch": 2, "world_sizes": list(range(8, 21, 1)), "lost_nodes": list(range(1, 2))},
    "gpt3_350M": {"microbatch": 32, "world_sizes": list(range(8, 21, 1)), "lost_nodes": list(range(1, 2))},

}

# 超时时间（秒）
TIMEOUT_SECONDS = 600
CUDA_MEMORY= 16945709056
OUT_DIR = "/workspace/Oobleck/important_data/nsdi/lost_nodes/"
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
    for world_size in config["world_sizes"]:
        for lost_nodes in config["lost_nodes"]:
            if not os.path.exists(f"/workspace/Oobleck/planning/pipeline_templates/{model}-{microbatch}-{world_size}-1.json"):
                continue
            command = COMMAND_TEMPLATE.format(model, microbatch, world_size, lost_nodes, OUT_DIR, CUDA_MEMORY)
            print(f"Running command: {command}")
            run_with_timeout(command, TIMEOUT_SECONDS)
            
            # exit()
