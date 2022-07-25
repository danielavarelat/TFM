#!/bin/bash

#SBATCH -J padding
#SBATCH -p high
#SBATCH -c 2
#SBATCH --mem-per-cpu=16G
#SBATCH -o %N.%J.out # STDOUT
#SBATCH -e %N.%j.err # STDERR

set -e

source ~/anaconda3/bin/activate "";
conda activate cellpose;
cd /homedtic/dvarela
python padding.py