import os
import threading
import time
import subprocess



MODELS = ["gpt3_350M", "gpt3_1_3B", "gpt3_2_7B","gpt3_6_7B" ]
MIN_WORLD_SIZE = 2
MAX_WORLD_SIZE = 3
WORLD_SIZE_INTERVAL = 1
MAX_MBS = 32
TIMEOUT_SECONDS = 600

NODE_IPS = ["172.31.11.113", "172.31.11.170"]
NODE_PORTS = ["2220"]
MASTER_IP = "172.31.11.113"
MASTER_PORT  = "60000"

MONITOR_INTERVAL = 15


def get_nodes_and_ports(world_size: int) -> tuple[str, str]:
    nodes = ""
    ports = ""
    # ports = []

    node_nums = len(NODE_IPS)

    batch = world_size // node_nums
    
    if world_size % node_nums != 0:
        batch += 1
    port_idx = 0
    i = 0
    for node_idx in range(node_nums):
        for port_idx in range(batch):
            nodes += NODE_IPS[node_idx] + " "
            ports += NODE_PORTS[port_idx] + " "
            # ports.append(NODE_PORTS[port_idx])
            i += 1
            if i == world_size:
                return (nodes, ports)
    return (nodes, ports)

def run_job(model: str, world_size: int, mbs: int) :
    config_file = f'./examples/tmp-{model}-{world_size}-{mbs}.yaml'
    os.environ['world_size'] = str(world_size)
    os.environ['mbs'] = str(mbs)
    template_file = f'./examples/{model}.template.yaml'
    with open(template_file) as f_template:
        template_content = f_template.read()
    config_content = os.popen(f'echo "{template_content}" | envsubst').read()
    with open(config_file, 'w') as f_config:
        f_config.write(config_content)
    os.environ['world_size'] = ""
    os.environ['mbs'] = ""
    nodes, ports = get_nodes_and_ports(world_size)
    # 运行python命令
    command = f"python -m oobleck.run --config_path {config_file} --node_ips {nodes.strip()} --node_ports {ports.strip()} --master_ip {MASTER_IP} --master_port {MASTER_PORT}"

    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result

def monitor_logs():
    '''
    monitor logs for TIMEOUT_SECONDS. Check whether error occurs or not. 
    '''
    log_dir = './tmp/logs'
    iters = TIMEOUT_SECONDS // MONITOR_INTERVAL + 1
    for i in range(iters):
        time.sleep(MONITOR_INTERVAL)
        # 找到最新的文件夹
        dirs = [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
        if not dirs:
            continue
        latest_dir = max(dirs, key=os.path.getmtime)
        if i == 0:
            print(f"monitoring log {latest_dir}")
        # 检查最新文件夹中的文件
        files = [os.path.join(latest_dir, f) for f in os.listdir(latest_dir) if os.path.isfile(os.path.join(latest_dir, f))]
        cuda_oom = False
        runtime_error = False
        for file in files:
            with open(file, 'r') as f:
                content = f.read()
                if 'CUDA out of memory' in content:
                    cuda_oom = True
                elif 'RuntimeError' in content:
                    runtime_error = True
                elif 'Training is done.' in content:
                    return 0
        if cuda_oom:
            return -1
        elif runtime_error:
            return -2
    return 0

def kill_processes():
    subprocess.run("./exp/kill.sh", shell=True, capture_output=True, text=True)

for model in MODELS:
    mbs = MAX_MBS
    for world_size in range(MAX_WORLD_SIZE, MIN_WORLD_SIZE - 1, -WORLD_SIZE_INTERVAL):
        while mbs > 0:
            print(f"start exp: {model}-{world_size}-{mbs}.")
            master_cmd = f"python -m oobleck.elastic.master  --ip {MASTER_IP} --port {MASTER_PORT}"
            master_proc = subprocess.Popen(master_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(5)
            job_proc = run_job(model, world_size, mbs)
            if job_proc.returncode != 0:
                print(f"finish exp: {model}-{world_size}-{mbs}. run job error. stdout: {job_proc.stdout}. stderr: {job_proc.stderr}")
                exit()
            
            res = monitor_logs()
            kill_processes()
            time.sleep(5)
            if res == 0:
                print(f"finish exp: {model}-{world_size}-{mbs} success.")
                break
            elif res == -1:
                mbs = mbs // 2
                print(f"finish exp: {model}-{world_size}-{mbs} CUDA OOM.")
                continue
            else:
                print(f"finish exp: {model}-{world_size}-{mbs} Runtime error.")
                break


