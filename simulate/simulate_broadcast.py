import asyncio
import os
import aiofiles
import asyncssh
import subprocess

model_configs = {
    "gpt3_1_3B": {"microbatch": 4, "world_sizes": list(range(9, 17))},
    "gpt3_2_7B": {"microbatch": 4, "world_sizes": list(range(10, 17))},
    # "gpt3_350M": {"microbatch": 8, "world_sizes": list(range(8, 16))},
}


async def run_command_on_node(node, command, label):
    output_file = f"/workspace/Oobleck/tmp/simulate_broadcast_logs/{label}/"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_file = f"{output_file}/{node}.log"

    # command = "date && sleep 10s && date"
    # Execute the command remotely
    try:
        async with asyncssh.connect(node, port=2222, known_hosts=None) as conn:
            async with aiofiles.open(
                output_file, "w"
            ) as log_file, conn.create_process(
                command,
                term_type="xterm",
            ) as process:
                print(f"Agent {node} output will be written at {output_file}")
                async for data in process.stdout:
                    await log_file.write(data)
                    await log_file.flush()
            print(f"run {command} on node {node}") 
    except (OSError, asyncssh.Error) as exc:
        print(f"Error connecting to {node}: {exc}")


async def run_model_tasks(nodes, layer_file, label):
    # Command template
    print(f"{layer_file} broadcast test begin")
    master_addr = "172.21.0.42"
    master_port = 10078
    command_template = '/bin/bash -ic "conda run --no-capture-output -n oobleck python /workspace/Oobleck/simulate/broadcast_test.py --master-ip {}  --master-port {} --node-rank {} --layer-file {} --gpus-per-node 4 --num-nodes 4"'
    # Create tasks for running commands on nodes
    tasks = []
    for node_rank, node in enumerate(nodes):
        command = command_template.format(
            master_addr, master_port, node_rank, layer_file
        )
        print(f"run command {command} on node {node}")
        task = asyncio.create_task(run_command_on_node(node, command, label))
        tasks.append(task)

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)
    print(f"All tasks for {layer_file} completed.")


async def main():
    nodes = ["172.21.0.42", "172.21.0.46", "172.21.0.90", "172.21.0.92"]
    label = "gpt3_2_7B-4-16"
    layer_file = f"/workspace/Oobleck/important_data/lost/{label}.json"
    await run_model_tasks(nodes, layer_file, label)
    exit()
    for model, config in model_configs.items():
        microbatch = config["microbatch"]
        for world_size in config["world_sizes"]:
            label = f"{model}-{microbatch}-{world_size}"
            layer_file = f"/workspace/Oobleck/important_data/lost/{label}.json"
            await run_model_tasks(nodes, layer_file, label)
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())
