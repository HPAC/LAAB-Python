import jax
import jax.numpy as jnp
import os
import time


@jax.jit
def actual(A,B):
    ret = A@B
    return ret

@jax.jit
def jnp_matmul(A,B):
    ret = jnp.matmul(A,B)
    return ret


if __name__ == "__main__":

    jax.config.update('jax_platform_name', 'cpu')

    #Problem size
    N = int(os.environ.get("LAAB_N", 3000))
    REPS = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = jnp.float32

    key = jax.random.PRNGKey(0)
    A = jnp.tril(jax.random.normal(key, [N, N], dtype=DTYPE))
    B = jax.random.normal(key, [N, N], dtype=DTYPE)
   

    for i in range(REPS):
        start = time.perf_counter()
        ret = actual(A,B).block_until_ready()
        end = time.perf_counter()
        elapsed_actual = end-start
        
        start = time.perf_counter()
        ret = jnp_matmul(A,B).block_until_ready()
        end = time.perf_counter()
        elapsed_matmul = end-start
        
        
        print("[LAAB] Jax | mp_trmm | actual={:.5f} s | jnp_matmul={:.5f} s".format(elapsed_actual, elapsed_matmul))  
    

