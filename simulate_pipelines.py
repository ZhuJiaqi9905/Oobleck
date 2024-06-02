import subprocess
import itertools
import threading
import time

# 定义模型、microbatch 和 worldsize 的参数列表
models = ["gpt3_1_3B", "gpt3_2_7B", "gpt3_6_7B", "gpt3_350M"]

model_configs = {
    "gpt3_1_3B": {"microbatch": 4, "world_sizes": list(range(8, 9))},
    "gpt3_2_7B": {"microbatch": 4, "world_sizes": list(range(8, 9))},
    "gpt3_6_7B": {"microbatch": 2, "world_sizes": list(range(10, 9))},
    "gpt3_350M": {"microbatch": 8, "world_sizes": list(range(8, 9))},
}

# 超时时间（秒）
timeout_seconds = 600

# 定义要运行的命令模板
pipeline_command_template = (
    "python oobleck_simulate.py --model {} --microbatch {} --worldsize {}"
)
lost_command_template = (
    "python oobleck_simulate.py --model {} --microbatch {} --worldsize {}  --lost-nodes {}"
)

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
for model, config in model_configs.items():
    microbatch = config["microbatch"]
    for world_size in config["world_sizes"]:
        command = pipeline_command_template.format(model, microbatch, world_size)
        print(f"Running command: {command}")
        run_with_timeout(command, timeout_seconds)

#for model, config in model_configs.items():
#    microbatch = config["microbatch"]
#    for world_size in config["world_sizes"]:
#        command = lost_command_template.format(model, microbatch, world_size, 1)
#        print(f"Running command: {command}")
#        run_with_timeout(command, timeout_seconds)
