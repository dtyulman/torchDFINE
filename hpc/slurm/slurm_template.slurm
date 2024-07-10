#!/bin/bash 

# inputs to slurm (should be exported as enviroment variables)
: "${Q:=shanechi}"
: "${TIME:=72:00:00}"
: "${MEM:=32GB}"

: "${log_dir:=/scratch2/eerturk/logs}"
: "${log_file:=log.txt}"
: "${code_dir:=/scratch2/ebilgin/torch-multimodal/runners}"
: "${command:=python}"

# For discovery
module purge
module load gcc/8.3.0
module load cuda/10.0.130
module load cudnn/7.5.1-10.1

# mkdir -p $log_dir
# export MPLCONFIGDIR=/scratch2/eerturk/python-cache/mpl-cache
export GIT_PYTHON_REFRESH=quiet

# ${command}| tee -a ${log_dir}/${log_file}
${command}