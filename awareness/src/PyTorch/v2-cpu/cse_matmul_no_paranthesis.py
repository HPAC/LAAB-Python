import torch
import os
import time

@torch.jit.script
def actual(A,B):
    ret = torch.t(torch.t(A)@B)@torch.t(A)@B    
    return ret

@torch.jit.script
def optimized(A,B):
    tmp = torch.t(A)@B
    ret = torch.t(tmp)@tmp
    return ret

if __name__ == "__main__":

    #Sets the number of threads used for intraop parallelism on CPU.
    torch.set_num_threads(1)

    #Problem size
    n = int(os.environ.get("LAAB_N", 3000))
    reps = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = torch.float32

    A = torch.randn([n, n], dtype=DTYPE)
    B = torch.randn([n, n], dtype=DTYPE)


    for i in range(reps):
        start = time.perf_counter()
        ret = actual(A,B)
        end = time.perf_counter()
        elapsed_actual = end-start

        start = time.perf_counter()
        ret = optimized(A,B)
        end = time.perf_counter()
        elapsed_optimized = end-start


        print("[LAAB] PyTorch | cse_matmul_no_paranthesis | optimized={:.5f} s | actual={:.5f} s".format(elapsed_optimized, elapsed_actual))  
