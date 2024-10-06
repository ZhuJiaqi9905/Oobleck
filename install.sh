#!bash
# # ipopt
# sudo apt-get install gcc g++ gfortran git patch wget pkg-config liblapack-dev libmetis-dev

# cd tmp

# git clone https://github.com/coin-or-tools/ThirdParty-ASL.git
# cd ThirdParty-ASL
# ./get.ASL
# ./configure
# make
# sudo make install
# cd ..

# git clone https://github.com/coin-or-tools/ThirdParty-Mumps.git
# cd ThirdParty-Mumps
# ./get.Mumps
# ./configure
# make
# sudo make install
# cd ..

# git clone https://github.com/coin-or/Ipopt.git
# cd Ipopt
# mkdir build
# cd build
# ../configure
# make
# sudo make install
# cd ../..

# oobleck env
CONDA_BASE=$(conda info --base)

# conda env create -f environment.yml

source ${CONDA_BASE}/etc/profile.d/conda.sh
conda create -n oobleck python==3.10.0

conda activate oobleck
conda install cmake==3.27.8
conda install pyomo==6.6.2
conda install glpk==5.0
conda install ipopt==3.14.13
# pip install -r requirements.txt

conda clean --yes --all

conda pack -n oobleck

# install gcc-13
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-13 g++-13 -y
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 13 --slave /usr/bin/g++ g++ /usr/bin/g++-13

# 修改头文件中的oneapi