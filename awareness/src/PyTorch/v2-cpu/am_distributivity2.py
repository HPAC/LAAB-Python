import torch
import os
import time


@torch.jit.script
def optimized(A,H,x):
    ret = A@x - torch.t(H)@(H@x)
    return ret

@torch.jit.script
def actual(A,H,x):
    ret = (A - torch.t(H)@H)@x
    return ret

if __name__ == "__main__":

    #Sets the number of threads used for intraop parallelism on CPU.
    torch.set_num_threads(1)

    #Problem size
    n = int(os.environ.get("LAAB_N", 3000))
    reps = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = torch.float32

    A = torch.randn([n, n], dtype=DTYPE)
    H = torch.randn([n, n], dtype=DTYPE)
    x = torch.randn([n, 1], dtype=DTYPE)

    for i in range(reps):
        start = time.perf_counter()
        ret1 = actual(A,H,x)
        end = time.perf_counter()
        elapsed_actual = end-start

        start = time.perf_counter()
        ret1 = optimized(A,H,x)
        end = time.perf_counter()
        elapsed_optimized = end-start    
        
        print("[LAAB] PyTorch | am_distributivity2 | optimized={:.5f} s | actual={:.5f} s".format(elapsed_optimized,elapsed_actual))

