import torch
import os
import time
import numpy as np
from scipy import linalg
from scipy.sparse import diags


def optimized(A,B):
    main = np.diag(A)
    lower = np.diag(A,-1)
    upper = np.diag(A,1)
    Ax = diags([lower, main, upper], offsets=[-1, 0, 1], format='csr')
    ret = Ax @ B
    return ret

@torch.jit.script
def actual(A,B):
    ret = A@B
    #ret = torch.einsum('ij,jk->ik',A,B)
    return ret

@torch.jit.script
def linalg_matmul(A,B):
    ret = torch.linalg.matmul(A,B)
    #ret = torch.linalg.tridiagonal_matmul(A, B, diagonals_format='matrix')
    return ret


if __name__ == "__main__":


    #Sets the number of threads used for intraop parallelism on CPU.
    torch.set_num_threads(1)

    #Problem size
    N = int(os.environ.get("LAAB_N", 3000))
    REPS = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = torch.float32

    A = torch.randn([N, N], dtype=DTYPE)
    A = torch.diag(torch.diag(A,1),1)+torch.diag(torch.diag(A))+torch.diag(torch.diag(A,-1),-1)
    B = torch.randn([N, N], dtype=DTYPE)
   
    DTYPE = np.float32
    A_opt = np.random.randn(N,N).astype(DTYPE)
    A_opt = np.diag(np.diag(A_opt,1),1)+np.diag(np.diag(A_opt))+np.diag(np.diag(A_opt,-1),-1)
    A_opt = A_opt.ravel(order='F').reshape(A_opt.shape, order='F')
    B_opt = np.random.randn(N,N).astype(DTYPE)
    B_opt = B_opt.ravel(order='F').reshape(B_opt.shape, order='F')


    for i in range(REPS):
        start = time.perf_counter()
        ret = actual(A,B)
        end = time.perf_counter()
        elapsed_actual = end-start
        
        start = time.perf_counter()
        ret = linalg_matmul(A,B)
        end = time.perf_counter()
        elapsed_matmul = end-start
        
        start = time.perf_counter()
        ret = optimized(A_opt,B_opt)
        end = time.perf_counter()
        elapsed_optimized = end-start 
        
        print("[LAAB] PyTorch | mp_tridiag | optimized={:.5f} s | actual={:.5f} s | linalg_matmul={:.5f} s".format(elapsed_optimized, elapsed_actual, elapsed_matmul))  
    

