import jax
import jax.numpy as jnp
import os
import time

@jax.jit
def ref_positive(H,x,y):
    ret = (jnp.transpose(H)@y)@(jnp.transpose(x)@H) 
    return ret

@jax.jit
def ref_negative(H,x,y):
    # evaluates from right-to-left
    tmp1 = jnp.transpose(H)@y
    tmp2 = tmp1@jnp.transpose(x)
    ret = tmp2@H
    return ret

@jax.jit
def operator(H,x,y):
    ret = jnp.transpose(H)@y@jnp.transpose(x)@H 
    return ret

@jax.jit
def linalg_multidot(H,x,y):
    ret = jnp.linalg.multi_dot([jnp.transpose(H), y, jnp.transpose(x), H]) 
    return ret


if __name__ == "__main__":
    
    exp_name = os.path.basename(__file__).split(".")[0]
    
    jax.config.update('jax_platform_name', 'cpu')
    
    #Problem size
    n = int(os.environ.get("LAAB_N", 3000))
    reps = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = jnp.float32
    
    key = jax.random.PRNGKey(0)
    H = jax.random.normal(key, [n, n], dtype=DTYPE)
    x = jax.random.normal(key, [n, 1], dtype=DTYPE)
    y = jax.random.normal(key, [n, 1], dtype=DTYPE)



    for i in range(reps):
        #cache scrub 300MB
        _ = bytearray(300*1024*1024); _[:] = b'0'
        
        start = time.perf_counter()
        ret = operator(H,x,y).block_until_ready()
        end = time.perf_counter()
        
        elapsed_operator = end-start
        
        start = time.perf_counter()
        ret = linalg_multidot(H,x,y).block_until_ready()
        end = time.perf_counter()
        
        elapsed_multidot = end-start
        
        start = time.perf_counter()
        ret = ref_positive(H,x,y).block_until_ready()
        end = time.perf_counter()
        
        elapsed_ref_positive = end-start
        
        start = time.perf_counter()
        ret = ref_negative(H,x,y).block_until_ready()
        end = time.perf_counter()
        
        elapsed_ref_negative = end-start
        
        print("[LAAB] Jax | {} | ref_positive={:.5f} s | operator={:.5f} s | linalg_multidot={:.5f} s | ref_negative={:.5f} s".format(exp_name, elapsed_ref_positive,
                                                                                                                                                  elapsed_operator,
                                                                                                                                                  elapsed_multidot,
                                                                                                                                                  elapsed_ref_negative))
