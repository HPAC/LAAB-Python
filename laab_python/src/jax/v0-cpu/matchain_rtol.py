import jax
import jax.numpy as jnp
import os
import time

@jax.jit
def optimized(H,x):
    ret = jnp.transpose(H)@(H@x) 
    return ret

@jax.jit
def actual(H,x):
    ret = jnp.transpose(H)@H@x 
    return ret

@jax.jit
def linalg_multidot(H,x):
    ret = jnp.linalg.multi_dot([jnp.transpose(H), H, x]) 
    return ret


if __name__ == "__main__":
    
    jax.config.update('jax_platform_name', 'cpu')
    
    #Problem size
    n = int(os.environ.get("LAAB_N", 3000))
    reps = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = jnp.float32

    key = jax.random.PRNGKey(0)
    H = jax.random.normal(key, [n, n], dtype=DTYPE)
    x = jax.random.normal(key, [n, 1], dtype=DTYPE)



    for i in range(reps):
        start = time.perf_counter()
        ret = actual(H,x).block_until_ready()
        end = time.perf_counter()
        
        elapsed_actual = end-start
        
        start = time.perf_counter()
        ret = linalg_multidot(H,x).block_until_ready()
        end = time.perf_counter()
        
        elapsed_multidot = end-start
        
        start = time.perf_counter()
        ret = optimized(H,x).block_until_ready()
        end = time.perf_counter()
        
        elapsed_optimized = end-start
        
        print("[LAAB] Jax | matchain_rtol | optimized={:.5f} s | actual={:.5f} s | linalg_multidot={:.5f} s".format(elapsed_optimized,elapsed_actual,elapsed_multidot))
