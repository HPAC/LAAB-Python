import torch
import os
import time


@torch.jit.script
def operator(A):
    ret = A@torch.t(A)
    return ret

@torch.jit.script
def linalg_matmul(A):
    ret = torch.linalg.matmul(A,torch.t(A))
    return ret


if __name__ == "__main__":


    #Sets the number of threads used for intraop parallelism on CPU.
    THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))
    torch.set_num_threads(THREADS)

    #Problem size
    N = int(os.environ.get("LAAB_N", 3000))
    REPS = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = torch.float32

    A = torch.randn([N, N], dtype=DTYPE)


    for i in range(REPS):
        #cache scrub 300MB
        _ = bytearray(300*1024*1024); _[:] = b'0'
        
        start = time.perf_counter()
        ret = operator(A)
        end = time.perf_counter()
        elapsed_operator = end-start
        
        start = time.perf_counter()
        ret = linalg_matmul(A)
        end = time.perf_counter()
        elapsed_matmul = end-start
        
        print("[LAAB] PyTorch | mp_syrk | operator={:.5f} s | linalg_matmul={:.5f} s | ref_negative=R+sgemm".format(elapsed_operator, elapsed_matmul))  
    

