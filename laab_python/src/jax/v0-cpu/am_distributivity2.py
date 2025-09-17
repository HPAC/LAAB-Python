import jax
import jax.numpy as jnp
import os
import time


@jax.jit
def optimized(A,H,x):
    ret = A@x - jnp.transpose(H)@(H@x)
    return ret

@jax.jit
def actual(A,H,x):
    ret = (A - jnp.transpose(H)@H)@x
    return ret

if __name__ == "__main__":

    jax.config.update('jax_platform_name', 'cpu')
    
    #Problem size
    n = int(os.environ.get("LAAB_N", 3000))
    reps = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = jnp.float32

    key = jax.random.PRNGKey(0)
    A = jax.random.normal(key, [n, n], dtype=DTYPE)
    H = jax.random.normal(key, [n, n], dtype=DTYPE)
    x = jax.random.normal(key, [n, 1], dtype=DTYPE)

    for i in range(reps):
        #cache scrub 300MB
        _ = bytearray(300*1024*1024); _[:] = b'0'
        
        start = time.perf_counter()
        ret1 = actual(A,H,x).block_until_ready()
        end = time.perf_counter()
        elapsed_actual = end-start

        start = time.perf_counter()
        ret1 = optimized(A,H,x).block_until_ready()
        end = time.perf_counter()
        elapsed_optimized = end-start    
        
        print("[LAAB] Jax | am_distributivity2 | optimized={:.5f} s | actual={:.5f} s".format(elapsed_optimized,elapsed_actual))

