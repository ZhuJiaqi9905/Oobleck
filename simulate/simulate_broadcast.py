import asyncio
import os
import asyncssh
import subprocess

model_configs = {
    "gpt3_1_3B": {"microbatch": 4, "world_sizes": list(range(8, 17))},
    "gpt3_2_7B": {"microbatch": 4, "world_sizes": list(range(8, 17))},
    "gpt3_6_7B": {"microbatch": 2, "world_sizes": list(range(10, 17))},
    "gpt3_350M": {"microbatch": 8, "world_sizes": list(range(8, 16))},
}


async def run_command_on_node(node, command, label):
    output_file = f"/workspace/Oobleck/tmp/simulate_broadcast_logs/{label}/"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output_file = f"{output_file}/{node}.log"

    if node == "172.21.0.42":
        with open(output_file, "w") as f:
            result = subprocess.run(
                command, shell=True, stdout=f, stderr=subprocess.STDOUT
            )
            if result.returncode == 0:
                print(
                    f"Command on {node} executed successfully, output written to {output_file}"
                )
            else:
                print(
                    f"Error executing command on {node}, check {output_file} for details"
                )
    else:
        # Execute the command remotely
        try:
            async with asyncssh.connect(node, port=2222, known_hosts=None) as conn:
                result = await conn.run(command)
                with open(output_file, "w") as f:
                    f.write(result.stdout)
                    if result.stderr:
                        f.write(result.stderr)
                if result.exit_status == 0:
                    print(f"Command on {node} executed successfully")
                else:
                    print(f"Error executing command on {node}: {result.stderr}")
        except (OSError, asyncssh.Error) as exc:
            print(f"Error connecting to {node}: {exc}")


async def run_model_tasks(nodes, layer_file, label):
    # Command template
    print(f"{layer_file} broadcast test begin")
    master_addr = "172.21.0.42"
    master_port = 10078
    command_template = "python --master-addr {}  --master-port {} --node-rank {} --layer-file {} --gpus-per-node 4 --num-nodes 4"
    # Create tasks for running commands on nodes
    tasks = []
    for node_rank, node in enumerate(nodes):
        command = command_template.format(
            master_addr, master_port, node_rank, layer_file
        )
        task = asyncio.create_task(run_command_on_node(node, command, label))
        tasks.append(task)

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)
    print(f"All tasks for {layer_file} completed.")


async def main():
    nodes = ["172.21.0.42", "172.21.0.46", "172.21.0.90", "172.21.0.92"]

    for model, config in model_configs.items():
        microbatch = config["microbatch"]
        for world_size in config["world_sizes"]:
            label = f"{model}-{microbatch}-{world_size}"
            layer_file = f"/workspace/Oobleck/important_data/lost/{label}.json"
            await run_model_tasks(nodes, layer_file, label)
            exit

if __name__ == "__main__":
    asyncio.run(main())
