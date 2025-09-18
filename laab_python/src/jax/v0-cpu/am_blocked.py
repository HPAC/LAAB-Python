import jax
import jax.numpy as jnp
import os
import time

@jax.jit
def operator(A,B):
    ret = A@B
    return ret

@jax.jit
def jnp_matmul(A,B):
    ret = jnp.matmul(A,B)
    return ret

@jax.jit
def ref_positive(A1,A2,B1,B2):
    ret = jnp.concatenate((A1@B1, A2@B2),axis=0)
    return ret

def ref_negative(A,B):
    # no graph mode
    ret = A@B
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
        #cache scrub 300MB
        _ = bytearray(300*1024*1024); _[:] = b'0'
        
        start = time.perf_counter()
        ret1 = operator(A,B).block_until_ready()
        end = time.perf_counter()
        elapsed_operator = end-start
        
        start = time.perf_counter()
        ret = jnp_matmul(A,B).block_until_ready()
        end = time.perf_counter()
        elapsed_matmul = end-start
        
        start = time.perf_counter()
        ret1 = ref_positive(A1,A2,B1,B2).block_until_ready()
        end = time.perf_counter()
        elapsed_ref_positive = end-start
        
        start = time.perf_counter()
        ret1 = ref_negative(A,B).block_until_ready()
        end = time.perf_counter()
        elapsed_ref_negative = end-start    

        print("[LAAB] Jax | am_blocked | ref_positive={:.5f} s | operator={:.5f} s  | jnp_matmul={:.5f} s | ref_negative={:.5f} s".format(elapsed_ref_positive,
                                                                                                                                        elapsed_operator,
                                                                                                                                        elapsed_matmul,
                                                                                                                                        elapsed_ref_negative))
