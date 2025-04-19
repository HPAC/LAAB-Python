import torch
import os
import time

class bcolors:
    WARNING = '\033[93m'
    ENDC = '\033[0m'


@torch.jit.script
def gemm_torch(A,B):
    ret = A@B
    return ret

if __name__ == "__main__":

    #Check if MKL is enabled
    #print(bcolors.WARNING + "MKL Enabled : ", torch.backends.mkl.is_available(), bcolors.ENDC)


    #Sets the number of threads used for intraop parallelism on CPU.
    torch.set_num_threads(1)

    #Problem size
    N = os.environ["LAMP_N"] if "LAMP_N" in os.environ else 3000
    REPS = os.environ["LAMP_REPS"] if "LAMP_N" in os.environ else 1
    DTYPE = torch.float32

    A = torch.randn([N, N], dtype=DTYPE)
    B = torch.randn([N, N], dtype=DTYPE)


    for i in range(REPS):
        start = time.perf_counter()
        ret = gemm_torch(A,B)
        end = time.perf_counter()
        print("gemm_torch : ", end-start) 
    

