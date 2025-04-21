import os
import time
import jax
import jax.numpy as jnp
from jax import jit
# JIT-compiled matrix multiplication
@jit
def gemm_jax(A, B):
    return A @ B

if __name__ == "__main__":
    # Problem size
    N = int(os.environ["LAMP_N"]) if "LAMP_N" in os.environ else 3000
    REPS = int(os.environ["LAMP_REPS"]) if "LAMP_REPS" in os.environ else 1

    # Generate random matrices
    key = jax.random.PRNGKey(0)
    A = jax.random.normal(key, (N, N), dtype=jnp.float32)
    B = jax.random.normal(key + 1, (N, N), dtype=jnp.float32)

    # Warm-up run to trigger JIT compilation
    #gemm_jax(A, B).block_until_ready()

    for i in range(REPS):
        start = time.perf_counter()
        ret = gemm_jax(A, B)
        end = time.perf_counter()
        print("gemm_jax : ", end - start)

