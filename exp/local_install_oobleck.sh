#!/bin/bash

# 不能source ~/.bashrc。因为它发现不是shell就退出了
source ~/.bash_profile
# conda activate oobleck
conda init bash
source ~/.bash_profile
conda activate oobleck
conda env list
cd /workspace/Oobleck
pip install -e .