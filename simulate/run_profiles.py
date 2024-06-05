import threading
import time
import subprocess

model_configs = {
    "gpt3_1_3B": {"microbatch": 4, "world_sizes": list(range(8, 17))},
    "gpt3_2_7B": {"microbatch": 4, "world_sizes": list(range(8, 17))},
    "gpt3_6_7B": {"microbatch": 2, "world_sizes": list(range(10, 17))},
    "gpt3_350M": {"microbatch": 8, "world_sizes": list(range(8, 16))},
}

# 超时时间（秒）
timeout_seconds = 600

def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command '{command}' failed with error: {e}")


for model, config in model_configs.items():
    microbatch = config["microbatch"]
    for world_size in config["world_sizes"]:
        master_cmd = "./master.sh"
        master_proc = subprocess.Popen(master_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        timer = threading.Timer(timeout_seconds, master_proc.kill)
        try:
            timer.start()
            time.sleep(5)
            job_cmd = f"./job.sh {model} {world_size} {microbatch}"
            run_command(job_cmd)
            master_proc.communicate()
        finally:
            timer.cancel()
