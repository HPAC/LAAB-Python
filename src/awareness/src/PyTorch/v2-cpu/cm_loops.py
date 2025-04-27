import torch
import os
import time

@torch.jit.script
def actual(A,B,V,ret):
    for i in range(3):
        ret = A@B + torch.tensordot(V[i],torch.t(V[i]),dims=0)
    return ret

@torch.jit.script
def optimized(A,B,V,ret):
    tmp = A@B
    ret = torch.empty_like(A)
    for i in range(3):
        ret = tmp + torch.tensordot(V[i],torch.t(V[i]),dims=0)
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
    V = torch.randn([3, n], dtype=DTYPE)
    ret = torch.randn([n, n], dtype=DTYPE)

    for i in range(reps):
        start = time.perf_counter()
        ret1 = actual(A,B,V,ret)
        end = time.perf_counter()
        elapsed_actual = end-start
        
        start = time.perf_counter()
        ret1 = optimized(A,B,V,ret)
        end = time.perf_counter()
        elapsed_optimized = end-start

        print("[LAAB] PyTorch | cm_loops | optimized={:.5f} s | actual={:.5f} s".format(elapsed_optimized,elapsed_actual))
