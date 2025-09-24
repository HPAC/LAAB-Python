import jax
import jax.numpy as jnp
import os
import time


@jax.jit
def operator(A,B):
    ret = A@B
    return ret

@jax.jit
def linalg_matmul(A,B):
    ret = jnp.matmul(A,B)
    return ret


if __name__ == "__main__":

    exp_name = os.path.basename(__file__).split(".")[0]
    
    jax.config.update('jax_platform_name', 'cpu')

    #Problem size
    N = int(os.environ.get("LAAB_N", 3000))
    REPS = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = jnp.float32

    key = jax.random.PRNGKey(0)
    A = jnp.tril(jax.random.normal(key, [N, N], dtype=DTYPE))
    B = jax.random.normal(key, [N, N], dtype=DTYPE)
   

    for i in range(REPS):
        #cache scrub 300MB
        _ = bytearray(300*1024*1024); _[:] = b'0'
        
        start = time.perf_counter()
        ret = operator(A,B).block_until_ready()
        end = time.perf_counter()
        elapsed_operator = end-start
        
        start = time.perf_counter()
        ret = linalg_matmul(A,B).block_until_ready()
        end = time.perf_counter()
        elapsed_matmul = end-start
        
        
        print("[LAAB] Jax | {} | operator={:.5f} s | linalg_matmul={:.5f} s | ref_negative=R+mm_sgemm".format(exp_name, elapsed_operator, elapsed_matmul))  
    

