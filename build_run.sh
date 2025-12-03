#!/bin/bash

##################### SLURM (do not change) v  #####################
#SBATCH --export=ALL
#SBATCH --job-name="mt"
#SBATCH --nodes=1
#SBATCH --output="mt.%j.%N.out"
#SBATCH -t 00:25:00
##################### SLURM (do not change) ^  #####################

# Above are SLURM directives for job scheduling on a cluster,
export SLURM_CONF=/etc/slurm/slurm.conf


echo "----- Building -----"
# Do not change below, it is fixed for everyone
SHAREDDIR=/home/coe4sp4/

# Source Intel MKL environment
source /opt/intel/oneapi/setvars.sh --force

#cmake -S . -B $(pwd)/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=${SHAREDDIR}/libpfm4/ -DPROFILING_ENABLED=ON -DUSE_MKL=ON -DOPENMP=ON
cmake -S . -B $(pwd)/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=${SHAREDDIR}/libpfm4/ -DPROFILING_ENABLED=ON -DUSE_MKL=ON -DOPENMP=ON -DGPU_ENABLED=ON
cmake --build $(pwd)/build -- -j8



echo "---- Running CPU ----"

mkdir -p $(pwd)/logs
$(pwd)/build/vec_mul_vec --benchmark_out="$(pwd)/logs/mt.json" --benchmark_out_format=json 

