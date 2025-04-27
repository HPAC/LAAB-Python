import torch
import os
import time

@torch.jit.script
def actual(A,B):
    ret = (A@B)[2,2]
    #tmp = A@B
    #ret = tmp[2,2]
    return ret

@torch.jit.script
def optimized(A,B):
    ret = torch.dot(A[2],B[:,2])
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
        ret1 = actual(A,B)
        end = time.perf_counter()
        elapsed_actual = end-start
        
        start = time.perf_counter()
        ret1 = optimized(A,B)
        end = time.perf_counter()
        elapsed_optimized = end-start

        print("[LAAB] PyTorch | cm_partial_op_prod | optimized={:.5f} s | actual={:.5f} s".format(elapsed_optimized,elapsed_actual))
