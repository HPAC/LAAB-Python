import torch
import os
import time

@torch.jit.script
def ref_positive(H,x,y):
    ret = (torch.t(H)@y)@(torch.t(x)@H) 
    return ret

@torch.jit.script
def ref_negative(H,x,y):
    # evaluates from right-to-left
    tmp1 = torch.t(H)@y
    tmp2 = tmp1@torch.t(x)
    ret = tmp2@H
    return ret

@torch.jit.script
def operator(H,x,y):
    ret = torch.t(H)@y@torch.t(x)@H 
    return ret


@torch.jit.script
def linalg_multidot(H,x,y):
    ret = torch.linalg.multi_dot([torch.t(H), y, torch.t(x), H]) 
    return ret

if __name__ == "__main__":
    
    exp_name = os.path.basename(__file__).split(".")[0]
    
    #Sets the number of threads used for intraop parallelism on CPU.
    THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))
    torch.set_num_threads(THREADS)

    #Problem size
    n = int(os.environ.get("LAAB_N", 3000))
    reps = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = torch.float32
    
    H = torch.randn([n, n], dtype=DTYPE)
    x = torch.randn([n, 1], dtype=DTYPE)
    y = torch.randn([n, 1], dtype=DTYPE)



    for i in range(reps):
        #cache scrub 300MB
        _ = bytearray(300*1024*1024); _[:] = b'0'
        
        start = time.perf_counter()
        ret = operator(H,x,y)
        end = time.perf_counter()
        
        elapsed_operator = end-start
        
        start = time.perf_counter()
        ret = linalg_multidot(H,x,y)
        end = time.perf_counter()
        
        elapsed_multidot = end-start
        
        start = time.perf_counter()
        ret = ref_positive(H,x,y)
        end = time.perf_counter()
        
        elapsed_ref_positive = end-start
        
        start = time.perf_counter()
        ret = ref_negative(H,x,y)
        end = time.perf_counter()
        
        elapsed_ref_negative = end-start
        
        print("[LAAB] PyTorch | {} | ref_positive={:.5f} s | operator={:.5f} s | linalg_multidot={:.5f} s | ref_negative={:.5f} s".format(
            exp_name,
            elapsed_ref_positive,
            elapsed_operator,
            elapsed_multidot,
            elapsed_ref_negative))  
