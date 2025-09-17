import jax
import jax.numpy as jnp
import os
import time

@jax.jit
def actual(A,B,y):
    ret = jnp.transpose(A)@B@jnp.transpose(A)@B@y
    return ret

@jax.jit
def optimized(A,B,y):
    ret = jnp.transpose(A)@(B@(jnp.transpose(A)@(B@y)))
    return ret

if __name__ == "__main__":

    jax.config.update('jax_platform_name', 'cpu')
    
    #Problem size
    n = int(os.environ.get("LAAB_N", 3000))
    reps = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = jnp.float32

    key = jax.random.PRNGKey(0)
    A = jax.random.normal(key, (n, n), dtype=DTYPE)
    B = jax.random.normal(key, (n, n), dtype=DTYPE)
    y = jax.random.normal(key, (n, 1), dtype=DTYPE)


    for i in range(reps):
        #cache scrub 300MB
        _ = bytearray(300*1024*1024); _[:] = b'0'
        
        start = time.perf_counter()
        ret = actual(A,B,y).block_until_ready()
        end = time.perf_counter()
        elapsed_actual = end-start

        start = time.perf_counter()
        ret = optimized(A,B,y).block_until_ready()
        end = time.perf_counter()
        elapsed_optimized = end-start


        print("[LAAB] Jax | cse_suboptimal | optimized={:.5f} s | actual={:.5f} s".format(elapsed_optimized, elapsed_actual))  
