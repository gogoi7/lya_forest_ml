#!/bin/bash
#SBATCH -J galaxy_full
#SBATCH -A AST23010
#SBATCH -p vm-small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 16:00:00
#SBATCH -o galaxy_full_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=axg210023@utdallas.edu

set -euo pipefail
cd /scratch/08488/anta1997/EX_IllustrisTNG

model="${1:?Supply EX_0, EX_1, EX_2, or EX_3}"
snapnum="${2:-90}"
printf -v snap_tag '%03d' "$snapnum"

export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

/usr/bin/python -u extract_galaxy_spectra.py \
    --root /scratch/08488/anta1997/EX_IllustrisTNG \
    --model "$model" \
    --snapnum "$snapnum" \
    --rays "galaxy_rays_${snap_tag}.csv" \
    --output "galaxy_spectra/galaxy_full_${model}_${snap_tag}.hdf5"`