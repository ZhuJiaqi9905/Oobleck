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

source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate oobleck

# pip install -r requirements.txt

conda clean --yes --all

conda pack -n oobleck