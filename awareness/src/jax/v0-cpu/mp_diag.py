import jax
import jax.numpy as jnp
import os
import time
import numpy as np
from scipy import linalg
from scipy.sparse import diags

import psutil


def optimized(A,B):
    p = psutil.Process(os.getpid())
    print("Number of threads used (SciPy):", p.num_threads())
    main = np.diag(A)
    Ax = diags([main,], offsets=[0,], format='csr')
    ret = Ax @ B
    return ret

@jax.jit
def actual(A,B):
    p = psutil.Process(os.getpid())
    print("Number of threads used:", p.num_threads())
    ret = A@B
    return ret

@jax.jit
def jnp_matmul(A,B):
    ret = jnp.matmul(A,B)
    return ret


if __name__ == "__main__":

    #Problem size
    N = int(os.environ.get("LAAB_N", 8000))
    REPS = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = jnp.float32

    key = jax.random.PRNGKey(0)
    A = jax.random.normal(key, [N, N], dtype=DTYPE)
    A = jnp.diag(jnp.diag(A))
    B = jax.random.normal(key, [N, N], dtype=DTYPE)
   
    DTYPE = np.float32
    A_opt = np.random.randn(N,N).astype(DTYPE)
    A_opt = A_opt.ravel(order='F').reshape(A_opt.shape, order='F')
    B_opt = np.random.randn(N,N).astype(DTYPE)
    B_opt = B_opt.ravel(order='F').reshape(B_opt.shape, order='F')


    for i in range(REPS):
        start = time.perf_counter()
        ret = actual(A,B)
        end = time.perf_counter()
        elapsed_actual = end-start
        
        start = time.perf_counter()
        ret = jnp_matmul(A,B)
        end = time.perf_counter()
        elapsed_matmul = end-start
        
        start = time.perf_counter()
        ret = optimized(A_opt,B_opt)
        end = time.perf_counter()
        elapsed_optimized = end-start 
        
        print("[LAAB] Jax | mp_diag | optimized={:.5f} s | actual={:.5f} s | linalg_matmul={:.5f} s".format(elapsed_optimized, elapsed_actual, elapsed_matmul))  
    

