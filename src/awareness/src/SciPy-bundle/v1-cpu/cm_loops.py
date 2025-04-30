import jax
import jax.numpy as jnp
import os
import time

@jax.jit
def actual(A,B,V,ret):
    for i in range(3):
        ret = A@B + jnp.tensordot(V[i],jnp.transpose(V[i]),axes=0)
    return ret

@jax.jit
def optimized(A,B,V,ret):
    tmp = A@B
    for i in range(3):
        ret = tmp + jnp.tensordot(V[i],jnp.transpose(V[i]),axes=0)
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
    V = jax.random.normal(key, [3, n], dtype=DTYPE)
    ret = jax.random.normal(key, [n, n], dtype=DTYPE)

    for i in range(reps):
        start = time.perf_counter()
        ret1 = actual(A,B,V,ret)
        end = time.perf_counter()
        elapsed_actual = end-start
        
        start = time.perf_counter()
        ret1 = optimized(A,B,V,ret)
        end = time.perf_counter()
        elapsed_optimized = end-start

        print("[LAAB] Jax | cm_loops | optimized={:.5f} s | actual={:.5f} s".format(elapsed_optimized,elapsed_actual))
