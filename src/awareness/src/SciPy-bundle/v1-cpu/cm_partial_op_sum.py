import jax
import jax.numpy as jnp
import os
import time

@jax.jit
def actual(A,B):
    ret = (A+B)[2,2]
    return ret

@jax.jit
def optimized(A,B):
    ret = A[2,2]+B[2,2]
    return ret

if __name__ == "__main__":

    jax.config.update('jax_platform_name', 'cpu')
    
    #Problem size
    n = int(os.environ.get("LAAB_N", 3000))
    reps = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = jnp.float32

    key = jax.random.PRNGKey(0)    
    A = jax.random.normal(key, [n, n], dtype=DTYPE)
    B = jax.random.normal(key, [n, n], dtype=DTYPE)

    for i in range(reps):
        start = time.perf_counter()
        ret1 = actual(A,B)
        end = time.perf_counter()
        elapsed_actual = end-start
        
        start = time.perf_counter()
        ret1 = optimized(A,B)
        end = time.perf_counter()
        elapsed_optimized = end-start

        print("[LAAB] Jax | cm_partial_op_sum | optimized={:.8f} s | actual={:.8f} s".format(elapsed_optimized,elapsed_actual))
