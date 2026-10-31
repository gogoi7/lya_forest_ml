#!/bin/bash
#SBATCH -J galaxy_match
#SBATCH -A AST23010
#SBATCH -p vm-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:15:00
#SBATCH -o galaxy_match_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=axg210023@utdallas.edu

set -e
cd /scratch/08488/anta1997/EX_IllustrisTNG

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

/usr/bin/python -u galaxy_absorption.py \
    --task match \
    --root /scratch/08488/anta1997/EX_IllustrisTNG \
    --snapnum 90 \
    --sample ex0_sample_z0.csv \
    --output galaxy_matches_090.csv