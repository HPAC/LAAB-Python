import torch
import os
import time


@torch.jit.script
def operator(A,B):
    ret = A@B
    #ret = torch.einsum('ij,jk->ik',A,B)
    return ret

@torch.jit.script
def linalg_matmul(A,B):
    ret = torch.linalg.matmul(A,B)
    #ret = torch.linalg.tridiagonal_matmul(A, B, diagonals_format='matrix')
    return ret


if __name__ == "__main__":

    exp_name = os.path.basename(__file__).split(".")[0]
    
    #Sets the number of threads used for intraop parallelism on CPU.
    THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))
    torch.set_num_threads(THREADS)

    #Problem size
    N = int(os.environ.get("LAAB_N", 3000))
    REPS = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = torch.float32

    T = torch.randn([N, N], dtype=DTYPE)
    T = torch.diag(torch.diag(T,1),1)+torch.diag(torch.diag(T))+torch.diag(torch.diag(T,-1),-1)
    B = torch.randn([N, N], dtype=DTYPE)


    for i in range(REPS):
        #cache scrub 300MB
        _ = bytearray(300*1024*1024); _[:] = b'0'
        
        start = time.perf_counter()
        ret = operator(T,B)
        end = time.perf_counter()
        elapsed_operator = end-start
        
        start = time.perf_counter()
        ret = linalg_matmul(T,B)
        end = time.perf_counter()
        elapsed_matmul = end-start
        
        print("[LAAB] PyTorch | {} | operator={:.5f} s | linalg_matmul={:.5f} s | ref_negative=R+mm_sgemm".format(exp_name, elapsed_operator, elapsed_matmul))  
    

