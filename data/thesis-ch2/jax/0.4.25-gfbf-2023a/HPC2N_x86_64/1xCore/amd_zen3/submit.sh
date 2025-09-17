#!/bin/bash
#SBATCH --job-name=jax_1xcore
#SBATCH --account=hpc2n2025-096
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1           
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=a100:1          

ml purge
ml GCC/12.3.0 OpenMPI/4.1.5
ml jax/0.4.25

export LD_BLAS='-lopenblas'
export OMP_NUM_THREADS=1
export LAAB_REPS=10
export LAAB_N=3000
export LAAB_PREFIX='taskset -c 0'

git_root=$(git rev-parse --show-toplevel)
export SRC_DIR=$git_root/laab_python/src/jax/v0-cpu/
cp $SRC_DIR/config.json .


lscpu > data.txt
srun make --no-print-directory -C $SRC_DIR/  | tee -a data.txt




