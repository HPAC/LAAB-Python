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
    THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))
    torch.set_num_threads(THREADS)

    #Problem size
    n = int(os.environ.get("LAAB_N", 3000))
    reps = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = torch.float32

    A = torch.randn([n, n], dtype=DTYPE)
    H = torch.randn([n, n], dtype=DTYPE)
    x = torch.randn([n, 1], dtype=DTYPE)

    for i in range(reps):
        #cache scrub 300MB
        _ = bytearray(300*1024*1024); _[:] = b'0'
        
        start = time.perf_counter()
        ret1 = actual(A,H,x)
        end = time.perf_counter()
        elapsed_actual = end-start

        start = time.perf_counter()
        ret1 = optimized(A,H,x)
        end = time.perf_counter()
        elapsed_optimized = end-start    
        
        print("[LAAB] PyTorch | am_distributivity2 | optimized={:.5f} s | actual={:.5f} s".format(elapsed_optimized,elapsed_actual))

