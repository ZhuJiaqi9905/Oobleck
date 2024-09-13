
import os
import re
'''  
读取文件夹下的所有log文件，获取时间。
'''
def extract_total_time(line):
    match = re.search(r'\[Rank 0\] time \(ms\) \| step: (\d+\.\d+)', line)
    if match:
        return float(match.group(1))
    return None

def extract_compute_time(line):
    match = re.search(r'train step time: (\d+\.\d+)s', line)
    if match:
        return float(match.group(1))
    return None   

def get_times(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    total_times = [extract_total_time(line) for line in lines if extract_total_time(line) is not None]
    return total_times 

def calculate_average(times):
    if len(times) != 3:
        return None
    return sum(times) / len(times)

def main():
    logs_dir = './tmp/logs/'

    dir_names = []
    for dir_name in os.listdir(logs_dir):
        dir_names.append(dir_name)
    dir_names.sort()
    for dir_name in dir_names:
        file_path = os.path.join(logs_dir, dir_name, '172.21.0.42-2220.out')
        

        if os.path.exists(file_path):
            model_name = dir_name.split('-')[6]
        
            nodes = dir_name.split('-')[9]
            total_times = get_times(file_path)
            if len(total_times) > 1:
                print(f"Folder: {dir_name}. model: {model_name}. nodes: {nodes} .Average total time: {sum(total_times[1:]) / len(total_times[1:])}")
                # exit()

if __name__ == "__main__":
    main()
