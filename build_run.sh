#!/bin/bash

##################### SLURM (do not change) v  #####################
#SBATCH --export=ALL
#SBATCH --job-name="tut03"
#SBATCH --nodes=1
#SBATCH --output="tut03.%j.%N.out"
#SBATCH -t 00:15:00
##################### SLURM (do not change) ^  #####################
export SLURM_CONF=/etc/slurm/slurm.conf

echo "----- Building tut03 -----"
cmake -S . -B $(pwd)/build -DCMAKE_BUILD_TYPE=Release
cmake --build $(pwd)/build -- -j8
make -j8

echo "---- Running tut03 ----"
$(pwd)/build/tensor-core-demo --json tensor-core-demo_results.json

echo "---- Testing  ----"
$(pwd)/build/test/vec_op_test

 
echo "---- Profiling with Nsight Compute ----"
mkdir -p $HOME/tmp
export TMPDIR=$HOME/tmp
mkdir -p $(pwd)/logs
ncu --set full -o $(pwd)/logs/ncu_profile $(pwd)/build/tensor-core-demo --profile

echo "---- Extracting NCU results to CSV ----"
ncu --import $(pwd)/logs/ncu_profile.ncu-rep --csv > $(pwd)/logs/ncu_results.csv

echo "---- Plotting ----"
# create python virtual environment
python3 -m venv $(pwd)/venv
source $(pwd)/venv/bin/activate
pip install -r $(pwd)/script/requirements.txt
mkdir -p $(pwd)/plots

python3 $(pwd)/script/plot.py $(pwd)/logs/ncu_results.csv

