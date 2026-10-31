#!/bin/bash
#SBATCH -J galaxy_pilot
#SBATCH -A AST23010
#SBATCH -p vm-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -o galaxy_pilot_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=axg210023@utdallas.edu

set -e
cd /scratch/08488/anta1997/EX_IllustrisTNG

model="${1:?Supply EX_0, EX_1, EX_2, or EX_3}"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

/usr/bin/python -u extract_galaxy_spectra.py \
    --root /scratch/08488/anta1997/EX_IllustrisTNG \
    --model "$model" \
    --snapnum 90 \
    --rays galaxy_rays_pilot_090.csv \
    --output "galaxy_spectra/galaxy_pilot_${model}_090.hdf5"