import jax
import jax.numpy as jnp
import os
import time


@jax.jit
def actual(A):
    ret = A@jnp.transpose(A)
    return ret

@jax.jit
def jnp_matmul(A):
    ret = jnp.matmul(A,jnp.transpose(A))
    return ret


if __name__ == "__main__":

    jax.config.update('jax_platform_name', 'cpu')
    
    #Problem size
    N = int(os.environ.get("LAAB_N", 3000))
    REPS = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = jnp.float32

    key = jax.random.PRNGKey(0)
    A = jax.random.normal(key, [N, N], dtype=DTYPE)

    # from threadpoolctl import threadpool_info
    # import pprint

    # pprint.pprint(threadpool_info())

    for i in range(REPS):
        #cache scrub 300MB
        _ = bytearray(300*1024*1024); _[:] = b'0'
        
        start = time.perf_counter()
        ret = actual(A).block_until_ready()
        end = time.perf_counter()
        elapsed_actual = end-start
        
        start = time.perf_counter()
        ret = jnp_matmul(A).block_until_ready()
        end = time.perf_counter()
        elapsed_matmul = end-start
        
        
        print("[LAAB] Jax | mp_syrk | actual={:.5f} s | jnp_matmul={:.5f} s".format(elapsed_actual, elapsed_matmul))  
    

