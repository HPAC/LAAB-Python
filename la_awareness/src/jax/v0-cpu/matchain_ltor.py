import jax
import jax.numpy as jnp
import os
import time


@jax.jit
def optimized(H,y):
    ret = (jnp.transpose(y)@jnp.transpose(H))@H 
    return ret

@jax.jit
def actual(H,y):
    ret = jnp.transpose(y)@jnp.transpose(H)@H 
    return ret

@jax.jit
def linalg_multidot(H,y):
    ret = jnp.linalg.multi_dot([jnp.transpose(y), jnp.transpose(H), H]) 
    return ret

if __name__ == "__main__":
    
    jax.config.update('jax_platform_name', 'cpu')
    
    #Problem size
    n = int(os.environ.get("LAAB_N", 3000))
    reps = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = jnp.float32
    
    key = jax.random.PRNGKey(0)
    H = jax.random.normal(key, [n, n], dtype=DTYPE)
    y = jax.random.normal(key, [n, 1], dtype=DTYPE)


    for i in range(reps):
        start = time.perf_counter()
        ret = actual(H,y)
        end = time.perf_counter()
        
        elapsed_actual = end-start
        
        start = time.perf_counter()
        ret = linalg_multidot(H,y)
        end = time.perf_counter()
        
        elapsed_multidot = end-start
        
        start = time.perf_counter()
        ret = optimized(H,y)
        end = time.perf_counter()
        
        elapsed_optimized = end-start
        
        print("[LAAB] Jax | matchain_ltor | optimized={:.5f} s | actual={:.5f} s | linalg_multidot={:.5f} s".format(elapsed_optimized,elapsed_actual,elapsed_multidot))
