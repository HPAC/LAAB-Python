#!/bin/bash
#SBATCH --job-name=pyt_vit_b16
#SBATCH --account=hpc2n2025-096
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --gpus-per-node=h100:2  
#SBATCH --output=srun.log      


git_root=$(git rev-parse --show-toplevel)
source $git_root/reports/PyTorch/2.1.2-foss-2023a/HPC2N_x86_64/env/activate.sh


export LAAB_LOG_DIR=$(pwd)/${SLURM_JOB_NAME}.${SLURM_JOB_ID}/

export LAAB_REPS=200

export SRC_DIR=$git_root/src/performance/PyTorch/v2/vit_b16

python $SRC_DIR/vit_b16_info.py

srun --overlap -n 1 --gpus=0 --cpus-per-task=1 monitor cunodeinfo nvpmon cpummon cpufreqmon &

export LAAB_GPU_BATCH=220
export LAAB_CPU_BATCH=1
srun -n 19 --wait=0 --multi-prog multi.conf --cpu-bind=cores 

srun --overlap -n 1 --gpus=0 --cpus-per-task=1 monitor stop

cp srun.log $LAAB_LOG_DIR/
cp submit.sh $LAAB_LOG_DIR/submit.sh
cp multi.conf $LAAB_LOG_DIR/ 