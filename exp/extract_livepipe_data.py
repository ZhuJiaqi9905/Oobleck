import os
import re
DIR = "/workspace/Oobleck/tmp/simulate_livepipe_logs"

if __name__ == "__main__":
    res = {}
    for file_dir in os.listdir(DIR):
        prefix = file_dir.split('.')[0].split('-')[-1]
        metadatas = prefix.split('_')
        model = metadatas[0]
        if model == "350M":
            ori_nodes = metadatas[2]
            curr_nodes = metadatas[4]
        else:
            model = metadatas[0] + "_" + metadatas[1]
            ori_nodes = metadatas[3]
            curr_nodes = metadatas[5]            
        with open(f"{DIR}/{file_dir}/172.21.0.42-2220.log") as f:
            for line in f:
                if '(GPU, Rank 0 | Time(averaged 2 times)' in line:
                    # 使用正则表达式提取所需信息
                    match = re.search(r'Time\(averaged 2 times\) = ([\d.]+) ms \| send_size = ([\d ]+) B \| recv_size = ([\d ]+) B', line)
                    if match:
                        # 提取的信息
                        time_taken = match.group(1)  # 716.29 ms
                        send_size = match.group(2)    # 0
                        recv_size = match.group(3)    # 0
                        res.setdefault(model, []).append((ori_nodes, curr_nodes, time_taken, send_size, recv_size))
                        break
    for l in res.values():
        l.sort(key=lambda x: x[0])
    for model, infos in res.items():
        for info in infos:
            print(f"{model} {info[0]} -> {info[1]}: {info[2]} ms | {info[3]} B | {info[4]} B")

    

                        