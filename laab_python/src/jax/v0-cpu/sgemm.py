import jax
import jax.numpy as jnp
import os
import time

#import psutil


@jax.jit
def actual(A,B):
    # p = psutil.Process(os.getpid())
    # print("Number of threads used:", p.num_threads())
    ret = A@B
    return ret

@jax.jit
def jnp_matmul(A,B):
    ret = jnp.matmul(A,B)
    return ret

if __name__ == "__main__":
    
    jax.config.update('jax_platform_name', 'cpu')
    # print(jax.devices())
    # print(jax.default_backend())
    
    #os.environ["XLA_FLAGS"] = ( "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1")
    
    #Problem size
    N = int(os.environ.get("LAAB_N", 3000))
    REPS = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = jnp.float32

    key = jax.random.PRNGKey(0)
    A = jax.random.normal(key, (N, N), dtype=DTYPE)
    B = jax.random.normal(key, (N, N), dtype=DTYPE)


    for i in range(REPS):
        #cache scrub 300MB
        _ = bytearray(300*1024*1024); _[:] = b'0'
        
        start = time.perf_counter()
        ret = actual(A,B).block_until_ready()
        end = time.perf_counter()
        elapsed_actual = end-start 
        
        start = time.perf_counter()
        ret = jnp_matmul(A,B).block_until_ready()
        end = time.perf_counter()
        elapsed_matmul = end-start
        
        print("[LAAB] Jax | sgemm | actual={:.5f} s | jnp_matmul={:.5f} s".format(elapsed_actual, elapsed_matmul)) 
