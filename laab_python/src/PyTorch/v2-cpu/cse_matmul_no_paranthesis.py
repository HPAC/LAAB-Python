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
    THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))
    torch.set_num_threads(THREADS)

    #Problem size
    n = int(os.environ.get("LAAB_N", 3000))
    reps = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = torch.float32

    A = torch.randn([n, n], dtype=DTYPE)
    B = torch.randn([n, n], dtype=DTYPE)


    for i in range(reps):
        #cache scrub 300MB
        _ = bytearray(300*1024*1024); _[:] = b'0'
        
        start = time.perf_counter()
        ret = actual(A,B)
        end = time.perf_counter()
        elapsed_actual = end-start

        start = time.perf_counter()
        ret = optimized(A,B)
        end = time.perf_counter()
        elapsed_optimized = end-start


        print("[LAAB] PyTorch | cse_matmul_no_paranthesis | optimized={:.5f} s | actual={:.5f} s".format(elapsed_optimized, elapsed_actual))  
