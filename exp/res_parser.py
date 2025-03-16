import os
import random
import re
from dateutil import parser as dateparser
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pprint


models = ['gpt3_1_3B', 'gpt3_2_7B', 'gpt3_6_7B', 'gpt3_13B']

iteration_time_parser = re.compile(r'train step time: (?P<iteration_time>\S+)s')
mb_size_parser = re.compile(r'mb size: (?P<mb_size>\d+)')

time_res = {}

def res_parser(file, model_size, node_num):
    if os.path.exists(file) is False:
        print(f'{file} does not exist')
        return
    # print(f'Parsing {file}')
    with open(file, 'r') as fp:
        iteration_time_avg = 0
        first = True
        steps = 0
        for line in fp.readlines():
            iteration_time_res = iteration_time_parser.search(line)
            mb_size_res = mb_size_parser.search(line)
            if iteration_time_res:
                if first:
                    first = False
                    continue
                steps += 1
                iteration_time_avg += float(iteration_time_res.group('iteration_time'))
            if mb_size_res:
                mbs = int(mb_size_res.group('mb_size'))
        if steps == 0:
            print(f'No iteration time found in {file}')
            return -1
        if time_res.get(model_size) is None:
            time_res[model_size] = {node_num:{mbs: {'iteration_time': iteration_time_avg / steps}}}
        else:
            if time_res[model_size].get(node_num) is None:
                time_res[model_size][node_num] = {mbs: {'iteration_time': iteration_time_avg / steps}}
            else:
                time_res[model_size][node_num][mbs] = {'iteration_time': iteration_time_avg / steps}
    return mbs
        
import csv

data_dir = 'important_data/2025-01/iter_time/'

with open('res.csv', 'w', newline='') as csvfile:
    fieldnames = ['models', 'node_num', 'mbs', 'iteration_time']
    writer = csv.writer(csvfile)
    writer.writerow(fieldnames)
    dirs = os.listdir(data_dir)
    dirs = sorted(dirs)
    for dir in dirs:
        model = dir.split('-')[-4]
        node_num = dir.split('-')[-1]
        mbs = res_parser(os.path.join(data_dir, dir, '172.31.34.170-2220.out'), model, node_num)
        if mbs > 0:
            writer.writerow([model, node_num, mbs, time_res[model][node_num][mbs]['iteration_time']])

pprint.pprint(time_res)