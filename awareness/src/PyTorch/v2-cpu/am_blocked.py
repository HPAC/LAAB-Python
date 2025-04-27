import torch
import os
import time

@torch.jit.script
def actual(A,B):
    ret = A@B
    return ret

@torch.jit.script
def linalg_matmul(A,B):
    ret = torch.linalg.matmul(A,B)
    return ret

@torch.jit.script
def optimized(A1,A2,B1,B2):
    ret = torch.cat((A1@B1, A2@B2),dim=0)
    return ret

if __name__ == "__main__":

    #Sets the number of threads used for intraop parallelism on CPU.
    torch.set_num_threads(1)

    #Problem size
    n = int(os.environ.get("LAAB_N", 3000))
    reps = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = torch.float32
    nb = int(n/2)

    A1 = torch.randn([nb, nb], dtype=DTYPE)
    A2 = torch.randn([nb, nb], dtype=DTYPE)
    A = torch.cat((torch.cat((A1, torch.zeros([nb,nb])) ,dim=1),torch.cat((torch.zeros([nb,nb]),A2) ,dim=1)),dim=0)

    B1 = torch.randn([nb, n], dtype=DTYPE)
    B2 = torch.randn([nb, n], dtype=DTYPE)
    B = torch.cat((B1,B2),dim=0)

    for i in range(reps):
        start = time.perf_counter()
        ret1 = actual(A,B)
        end = time.perf_counter()
        elapsed_actual = end-start
        
        start = time.perf_counter()
        ret = linalg_matmul(A,B)
        end = time.perf_counter()
        elapsed_matmul = end-start
        
        start = time.perf_counter()
        ret1 = optimized(A1,A2,B1,B2)
        end = time.perf_counter()
        elapsed_optimized = end-start    

        print("[LAAB] PyTorch | am_blocked | optimized={:.5f} s | actual={:.5f} s  | linalg_matmul={:.5f} s".format(elapsed_optimized, elapsed_actual, elapsed_matmul))
