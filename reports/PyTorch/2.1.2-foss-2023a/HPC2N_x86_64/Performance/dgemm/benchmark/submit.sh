#!/bin/bash
#SBATCH --job-name=pyt_performance
#SBATCH --account=hpc2n2025-096
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --gpus-per-node=h100:2  
#SBATCH --output=srun.log      

ml purge
ml GCC/12.3.0 OpenMPI/4.1.5
ml PyTorch/2.1.2-CUDA-12.1.1

export LD_BLAS='-lopenblas'

git_root=$(git rev-parse --show-toplevel)

export LAAB_BUILD_DIR=$(pwd)/build/
make -C $git_root/external/laab-utils/
source $LAAB_BUILD_DIR/set_env.sh

export LAAB_LOG_DIR=$(pwd)/${SLURM_JOB_NAME}.${SLURM_JOB_ID}/

export LAAB_REPS=200
export LAAB_CPU_N=1500
export LAAB_GPU_M=8192
export LAAB_GPU_K=12288
export LAAB_GPU_N=49152

export SRC_DIR=$git_root/src/performance/PyTorch/v2/gemm

python $SRC_DIR/dgemm_info.py dgemm_cpu $LAAB_CPU_N $LAAB_CPU_N $LAAB_CPU_N
python $SRC_DIR/dgemm_info.py dgemm_cuda $LAAB_GPU_M $LAAB_GPU_K $LAAB_GPU_N

srun --overlap -n 1 --gpus=0 --cpus-per-task=1 monitor cunodeinfo nvpmon cpummon cpufreqmon &
srun -n 19 --wait=0 --multi-prog multi.conf --cpu-bind=cores 
srun --overlap -n 1 --gpus=0 --cpus-per-task=1 monitor stop

cp srun.log $LAAB_LOG_DIR/
cp submit.sh $LAAB_LOG_DIR/submit.sh
cp multi.conf $LAAB_LOG_DIR/ 