import torch
import os
import time


@torch.jit.script
def actual(A,B,C):
    ret = A@B + A@C
    return ret

@torch.jit.script
def optimized(A,B,C):
    ret = A@(B+C)
    return ret

if __name__ == "__main__":

    #Sets the number of threads used for intraop parallelism on CPU.
    THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))
    torch.set_num_threads(THREADS)

    #Problem size
    n = int(os.environ.get("LAAB_N", 3000))
    reps = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = torch.float32

    A = torch.randn([n, n], dtype=DTYPE)
    B = torch.randn([n, n], dtype=DTYPE)
    C = torch.randn([n, n], dtype=DTYPE)

    for i in range(reps):
        start = time.perf_counter()
        ret1 = actual(A,B,C)
        end = time.perf_counter()
        elapsed_actual = end-start

        start = time.perf_counter()
        ret1 = optimized(A,B,C)
        end = time.perf_counter()
        elapsed_optimized = end-start    
        
        print("[LAAB] PyTorch | am_distributivity1 | optimized={:.5f} s | actual={:.5f} s".format(elapsed_optimized,elapsed_actual))

