import jax
import jax.numpy as jnp
import os
import time

@jax.jit
def operator(A,B):
    ret = (A+B)[2,2]
    return ret

@jax.jit
def ref_positive(A,B):
    ret = A[2,2]+B[2,2]
    return ret

@jax.jit
def ref_negative(A,B):
    ret = A+B
    return ret

if __name__ == "__main__":

    exp_name = os.path.basename(__file__).split(".")[0]
    
    jax.config.update('jax_platform_name', 'cpu')
    
    #Problem size
    n = int(os.environ.get("LAAB_N", 3000))
    reps = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = jnp.float32

    key = jax.random.PRNGKey(0)    
    A = jax.random.normal(key, [n, n], dtype=DTYPE)
    B = jax.random.normal(key, [n, n], dtype=DTYPE)

    for i in range(reps):
        #cache scrub 300MB
        _ = bytearray(300*1024*1024); _[:] = b'0'
        
        start = time.perf_counter()
        ret1 = operator(A,B).block_until_ready()
        end = time.perf_counter()
        elapsed_operator = end-start
        
        start = time.perf_counter()
        ret1 = ref_positive(A,B).block_until_ready()
        end = time.perf_counter()
        elapsed_ref_positive = end-start
        
        start = time.perf_counter()
        ret1 = ref_negative(A,B).block_until_ready()
        end = time.perf_counter()
        elapsed_ref_negative = end-start

        print("[LAAB] Jax | {} | ref_positive={:.8f} s | operator={:.8f} s | ref_negative={:.5f} s".format(exp_name, elapsed_ref_positive,
                                                                                                                          elapsed_operator,
                                                                                                                          elapsed_ref_negative))
