import torch
import os
import time


@torch.jit.script
def actual(A,B):
    ret = A@B
    return ret

if __name__ == "__main__":


    #Sets the number of threads used for intraop parallelism on CPU.
    torch.set_num_threads(1)

    #Problem size
    N = int(os.environ.get("LAAB_N", 3000))
    REPS = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = torch.float32

    A = torch.randn([N, N], dtype=DTYPE)
    B = torch.randn([N, N], dtype=DTYPE)


    for i in range(REPS):
        start = time.perf_counter()
        ret = actual(A,B)
        end = time.perf_counter()
        elapsed_actual = end-start 
        print("[LAAB] PyTorch | sgemm | actual={:.5f} s".format(elapsed_actual)) 