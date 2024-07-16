#!/bin/bash 

# inputs to slurm (should be exported as enviroment variables)
: "${Q:=shanechi}"
: "${TIME:=72:00:00}"
: "${MEM:=32GB}"

: "${log_dir:=/scratch1/ebilgin/logs}"
: "${log_file:=log.txt}"
: "${code_dir:=/scratch1/ebilgin/torchDFINE/runners}"
: "${env_dir:=/scratch1/ebilgin/torch/runners}"
: "${command:=python}"

# For discovery
module purge
module load gcc/8.3.0
module load cuda/10.0.130
module load cudnn/7.5.1-10.1
module load python/3.9.2

echo $env_dir
source $env_dir
# mkdir -p $log_dir
# export MPLCONFIGDIR=/scratch2/eerturk/python-cache/mpl-cache
export GIT_PYTHON_REFRESH=quiet

${command}| tee -a ${log_dir}/${log_file}