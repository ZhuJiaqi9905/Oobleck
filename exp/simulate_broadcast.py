import asyncio
import os
import aiofiles
import asyncssh
import subprocess
import time
import numpy as np

NODE_IPS = ["172.21.0.42", "172.21.0.46", "172.21.0.90", "172.21.0.91", "172.21.0.92"]
NODE_PORTS = [2220, 2221, 2222, 2223]


DIR = "/workspace/Oobleck/important_data/nsdi/lost_nodes/oobleck"
LOG_DIR = "/workspace/Oobleck/tmp/simulate_broadcast_logs/"
COMMAND_TEMPLATE = '''/bin/bash -ic "conda run --no-capture-output -n oobleck python /workspace/Oobleck/simulate/broadcast_test.py --master-ip 172.21.0.42  --master-port 10078 --gpus-per-node 1 --warmup-times 2 --repeat-times 10 --node-rank {} --layer-file {}"'''


def get_nodes_and_ports(world_size: int) -> tuple[list[str], list[int]]:
    '''
    根据world_size的大小来获得该使用哪些容器。
    '''
    nodes = []
    ports = []

    node_nums = len(NODE_IPS)

    batch = world_size // node_nums

    if world_size % node_nums != 0:
        batch += 1

    batch = 4
    port_idx = 0
    i = 0
    for node_idx in range(node_nums):
        for port_idx in range(batch):
            nodes.append(NODE_IPS[node_idx])
            ports.append(NODE_PORTS[port_idx])

            i += 1
            if i == world_size:
                return (nodes, ports)
    return (nodes, ports)

async def run_command_on_node(node_ip: str, node_port: int, command: str, prefix: str):
    """
    prefix: 存储log文件的文件夹前缀
    """

    output_file = (
        f"/{LOG_DIR}/{prefix}/"
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_file = f"{output_file}/{node_ip}-{node_port}.log"
    # Execute the command remotely
    try:
        async with asyncssh.connect(node_ip, port=node_port, known_hosts=None) as conn:
            async with aiofiles.open(output_file, "w") as log_file, conn.create_process(
                command,
                term_type="xterm",
            ) as process:
                # print(
                #     f"Agent {node_ip}-{node_port} output will be written at {output_file}"
                # )
                async for data in process.stdout:
                    await log_file.write(data)
                    await log_file.flush()
            print(f"run {command} on node {node_ip}-{node_port}")
    except (OSError, asyncssh.Error) as exc:
        print(f"Error connecting to {node_ip}-{node_port}: {exc}")


async def run_model_tasks(world_size: int, layer_file: str, prefix: str):
    '''
    prefix: 存储log文件的文件夹前缀
    '''
    print(f"{layer_file} broadcast test begin")
    # 会先根据world_size的大小来得到该使用哪些容器
    ips, ports = get_nodes_and_ports(world_size)
    # 在这些容器上启动任务
    tasks = []
    node_rank = 0
    current_time = time.localtime(time.time())
    current_time = time.strftime("%m-%d-%Y-%H-%M-%S", current_time)
    prefix = f"{current_time}-{prefix}"
    print(f"log prefix: {prefix}.ips: {ips}. ports: {ports}")
    assert(len(ips) == len(ports))
    for i in range(len(ips)):
        ip = ips[i]
        port = ports[i]
        command = COMMAND_TEMPLATE.format(node_rank, layer_file)
        node_rank += 1
        task = asyncio.create_task(run_command_on_node(ip, port, command, prefix))
        tasks.append(task)
        # print(f"node_rank: {node_rank}. command: {command}")
    # Wait for all tasks to complete
    await asyncio.gather(*tasks)
    print(f"{layer_file} test completed.")


async def main():
    # 遍历DIR文件夹下所有的文件。都是.json文件，并且文件命名方式为${MODEL}-{world_size}-{micro_batch_size}-{lost_nodes}.json

    files = ["gpt3_1_3B-20-16-5.json", "gpt3_1_3B-16-16-3.json", "gpt3_1_3B-15-16-2.json",
             "gpt3_2_7B-20-8-5.json", "gpt3_2_7B-16-8-3.json", "gpt3_2_7B-15-8-2.json",
             "gpt3_6_7B-20-2-5.json", "gpt3_6_7B-16-2-3.json", "gpt3_6_7B-15-2-2.json"
             ]
    for filename in os.listdir(DIR):
        if not filename.endswith(".json"):
            continue
        if not filename in files:
            continue
        prefix = filename.split('.')[0]
        metadatas = prefix.split('-')
        world_size = int(metadatas[1])
        # print(f"{prefix}, {world_size}")
        await run_model_tasks(world_size, f"{DIR}/{filename}", prefix)
        await asyncio.sleep(15)
if __name__ == "__main__":
    asyncio.run(main())
