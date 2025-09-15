import jax
import jax.numpy as jnp
import os
import time


@jax.jit
def actual(A,B,C):
    ret = A@B + A@C
    return ret

@jax.jit
def optimized(A,B,C):
    ret = A@(B+C)
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
    C = jax.random.normal(key, [n, n], dtype=DTYPE)

    for i in range(reps):
        start = time.perf_counter()
        ret1 = actual(A,B,C).block_until_ready()
        end = time.perf_counter()
        elapsed_actual = end-start

        start = time.perf_counter()
        ret1 = optimized(A,B,C).block_until_ready()
        end = time.perf_counter()
        elapsed_optimized = end-start    
        
        print("[LAAB] Jax | am_distributivity1 | optimized={:.5f} s | actual={:.5f} s".format(elapsed_optimized,elapsed_actual))

