import torch
import os
import time

@torch.jit.script
def operator(A,B,y):
    ret = torch.t(A)@B@torch.t(A)@B@y    
    return ret

@torch.jit.script
def ref_positive(A,B,y):
    ret = torch.t(A)@(B@(torch.t(A)@(B@y)))
    return ret

@torch.jit.script
def ref_negative(A,B,y):
    tmp1 = torch.t(A)@B
    tmp2  = tmp1@tmp1
    ret = tmp2@y
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

    A = torch.randn([n, n], dtype=DTYPE)
    B = torch.randn([n, n], dtype=DTYPE)
    y = torch.randn([n, 1], dtype=DTYPE)


    for i in range(reps):
        #cache scrub 300MB
        _ = bytearray(300*1024*1024); _[:] = b'0'
        
        start = time.perf_counter()
        ret = operator(A,B,y)
        end = time.perf_counter()
        elapsed_operator = end-start

        start = time.perf_counter()
        ret = ref_positive(A,B,y)
        end = time.perf_counter()
        elapsed_ref_positive = end-start
        
        start = time.perf_counter()
        ret = ref_negative(A,B,y)
        end = time.perf_counter()
        elapsed_ref_negative = end-start


        print("[LAAB] PyTorch | {} | ref_positive={:.5f} s | operator={:.5f} s | ref_negative={:.5f} s".format(exp_name, elapsed_ref_positive,
                                                                                                                           elapsed_operator,
                                                                                                                           elapsed_ref_negative))  
