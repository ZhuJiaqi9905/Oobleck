
import os
import re

def extract_time(line):
    match = re.search(r'\[Rank 0\] time \(ms\) \| step: (\d+\.\d+)', line)
    if match:
        return float(match.group(1))
    return None

def get_last_three_times(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    times = [extract_time(line) for line in lines if extract_time(line) is not None]
    return times[-3:]

def calculate_average(times):
    if len(times) != 3:
        return None
    return sum(times) / len(times)

def main():
    logs_dir = '/tmp/logs'
    average_times = []

    for root, dirs, files in os.walk(logs_dir):
        for dir_name in dirs:
            file_path = os.path.join(root, dir_name, '172.21.0.42-2220.out')
            if os.path.exists(file_path):
                last_three_times = get_last_three_times(file_path)
                if len(last_three_times) == 3:
                    average = calculate_average(last_three_times)
                    average_times.append(average)
                    print(f"Folder: {dir_name}, Average of last three times: {average}")

    if average_times:
        overall_average = sum(average_times) / len(average_times)
        print(f"Overall average of last three times from all folders: {overall_average}")

if __name__ == "__main__":
    main()
