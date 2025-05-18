#!/bin/bash
#SBATCH --job-name=jax_awareness
#SBATCH --account=hpc2n2025-096
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1             
#SBATCH --cpus-per-task=1          

ml purge
ml GCC/12.3.0 OpenMPI/4.1.5
ml jax/0.4.25

export LD_BLAS='-lopenblas'

git_root=$(git rev-parse --show-toplevel)
export SRC_DIR=$git_root/src/awareness
export LAAB_REPORTS_DIR=$(pwd)/..


lscpu
srun make -C $SRC_DIR/ jax_v0_cpu