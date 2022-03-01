#!/bin/bash

#SBATCH -J Cellpose_test
#SBATCH -p high
#SBATCH -c 8
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=16G
#SBATCH -o %N.%J.out # STDOUT
#SBATCH -e %N.%j.err # STDERR

set -e

ml CUDA/11.4.3
nvidia-smi
nvcc --version

ps -ef | grep slurm
uname -a >> /homedtic/dvarela/hehe.txt
source ~/anaconda3/bin/activate "";
conda activate cellpose;
cd /homedtic/dvarela/pretrained/cellpose
python run_cellpose.py