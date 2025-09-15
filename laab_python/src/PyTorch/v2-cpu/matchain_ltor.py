import torch
import os
import time


@torch.jit.script
def optimized(H,y):
    ret = (torch.t(y)@torch.t(H))@H 
    return ret

@torch.jit.script
def actual(H,y):
    ret = torch.t(y)@torch.t(H)@H 
    return ret

@torch.jit.script
def linalg_multidot(H,y):
    ret = torch.linalg.multi_dot([torch.t(y), torch.t(H), H]) 
    return ret

if __name__ == "__main__":
    
    #Sets the number of threads used for intraop parallelism on CPU.
    torch.set_num_threads(1)

    #Problem size
    n = int(os.environ.get("LAAB_N", 3000))
    reps = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = torch.float32
    
    H = torch.randn([n, n], dtype=DTYPE)
    y = torch.randn([n, 1], dtype=DTYPE)


    for i in range(reps):
        start = time.perf_counter()
        ret = actual(H,y)
        end = time.perf_counter()
        
        elapsed_actual = end-start
        
        start = time.perf_counter()
        ret = linalg_multidot(H,y)
        end = time.perf_counter()
        
        elapsed_multidot = end-start
        
        start = time.perf_counter()
        ret = optimized(H,y)
        end = time.perf_counter()
        
        elapsed_optimized = end-start
        
        print("[LAAB] PyTorch | matchain_ltor | optimized={:.5f} s | actual={:.5f} s | linalg_multidot={:.5f} s".format(elapsed_optimized,elapsed_actual,elapsed_multidot))  
