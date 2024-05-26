from functools import reduce
import json
import operator
model_configs = {
    "gpt3_1_3B": {"microbatch": 4, "world_sizes": list(range(9, 17))},
    "gpt3_2_7B": {"microbatch": 4, "world_sizes": list(range(10, 17))},
    # "gpt3_350M": {"microbatch": 8, "world_sizes": list(range(8, 16))},
}


def get_traffic(file_path: str) -> tuple[int, int] | None:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            # 读取并解析JSON文件
            data = json.load(file)
            layers = data["layers"]
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    traffic_map = {}
    for layer in layers:
        layer_size = 0
        for s in layer["sizes"]:
            layer_size += reduce(operator.mul, s, 1)
        layer_size *= 16
        if layer["ranks"][0] not in traffic_map:
            traffic_map[layer["ranks"][0]] = [layer_size, 0]
        else:
            traffic_map[layer["ranks"][0]][0] += layer_size
        for i in range(1, len(layer["ranks"])):
            if layer["ranks"][i] not in traffic_map:
                traffic_map[layer["ranks"][i]] = [0, layer_size]
            else:
                traffic_map[layer["ranks"][i]][1] += layer_size            
    # print(f"traffic_map: {traffic_map}")
    max_send_traffic = max([traffic[0] for traffic in traffic_map.values()])
    max_recv_traffic = max([traffic[1] for traffic in traffic_map.values()])
    return max_send_traffic / (2**30), max_recv_traffic / (2**30)

if __name__ == "__main__":
    for model, config in model_configs.items():
        microbatch = config["microbatch"]
        for world_size in config["world_sizes"]:
            label = f"{model}-{microbatch}-{world_size}"
            layer_file = f"/workspace/Oobleck/important_data/lost/{label}.json"
            (send_traffic, recv_traffic) = get_traffic(layer_file)
            print(f"{label}: send_traffic {send_traffic} GB, recv_traffic {recv_traffic} GB")