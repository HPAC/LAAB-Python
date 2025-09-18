import jax
import jax.numpy as jnp
import os
import time

@jax.jit
def operator(A,B,y):
    ret = jnp.transpose(A)@B@jnp.transpose(A)@B@y
    return ret

@jax.jit
def ref_positive(A,B,y):
    ret = jnp.transpose(A)@(B@(jnp.transpose(A)@(B@y)))
    return ret

@jax.jit
def ref_negative(A,B,y):
    tmp1 = jnp.transpose(A)@B
    tmp2  = tmp1@tmp1
    ret = tmp2@y
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
        ret = operator(A,B,y).block_until_ready()
        end = time.perf_counter()
        elapsed_operator = end-start

        start = time.perf_counter()
        ret = ref_positive(A,B,y).block_until_ready()
        end = time.perf_counter()
        elapsed_ref_positive = end-start


        print("[LAAB] Jax | cse_suboptimal | ref_positive={:.5f} s | operator={:.5f} s".format(elapsed_ref_positive, elapsed_operator))  
