#!/bin/bash

#SBATCH -J meshing
#SBATCH -p high
#SBATCH -c 8
#SBATCH --mem-per-cpu=16G
#SBATCH -o %N.%J.out # STDOUT
#SBATCH -e %N.%j.err # STDERR

set -e

source ~/anaconda3/bin/activate "";
conda activate porespy3;
cd /homedtic/dvarela/EXTRACTION
python mesh3d_from_df.py