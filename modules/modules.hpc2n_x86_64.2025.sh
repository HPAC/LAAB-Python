ml purge
ml GCC/12.3.0  OpenMPI/4.1.5
ml PyTorch/2.1.2
ml TensorFlow/2.15.1
ml jax/0.4.25

export LD_BLAS='-lopenblas'
export LAAB_PYT_CPU_RESULTS_DIR=reports/PyTorch/2.1.2-foss-2023a/HPC2N_x86_64
export LAAB_TF_CPU_RESULTS_DIR=reports/TensorFlow/2.15.1-foss-2023a/HPC2N_x86_64
export LAAB_JAX_CPU_RESULTS_DIR=reports/jax/0.4.25-gfbf-2023a/HPC2N_x86_64
export LAAB_SCIPY_CPU_RESULTS_DIR=reports/SciPy-bundle/2023.07-gfbf-2023a/HPC2N_x86_64

