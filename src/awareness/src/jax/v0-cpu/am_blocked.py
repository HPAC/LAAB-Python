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

@jax.jit
def optimized(A1,A2,B1,B2):
    ret = jnp.concatenate((A1@B1, A2@B2),axis=0)
    return ret

if __name__ == "__main__":

    jax.config.update('jax_platform_name', 'cpu')

    #Problem size
    n = int(os.environ.get("LAAB_N", 3000))
    reps = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = jnp.float32
    nb = int(n/2)

    key = jax.random.PRNGKey(0)
    A1 = jax.random.normal(key, [nb, nb], dtype=DTYPE)
    A2 = jax.random.normal(key, [nb, nb], dtype=DTYPE)
    A = jnp.concatenate((jnp.concatenate((A1, jnp.zeros([nb,nb])) ,axis=1),jnp.concatenate((jnp.zeros([nb,nb]),A2) ,axis=1)),axis=0)

    B1 = jax.random.normal(key, [nb, n], dtype=DTYPE)
    B2 = jax.random.normal(key, [nb, n], dtype=DTYPE)
    B = jnp.concatenate((B1,B2),axis=0)

    for i in range(reps):
        start = time.perf_counter()
        ret1 = actual(A,B).block_until_ready()
        end = time.perf_counter()
        elapsed_actual = end-start
        
        start = time.perf_counter()
        ret = jnp_matmul(A,B).block_until_ready()
        end = time.perf_counter()
        elapsed_matmul = end-start
        
        start = time.perf_counter()
        ret1 = optimized(A1,A2,B1,B2).block_until_ready()
        end = time.perf_counter()
        elapsed_optimized = end-start    

        print("[LAAB] Jax | am_blocked | optimized={:.5f} s | actual={:.5f} s  | jnp_matmul={:.5f} s".format(elapsed_optimized, elapsed_actual, elapsed_matmul))
