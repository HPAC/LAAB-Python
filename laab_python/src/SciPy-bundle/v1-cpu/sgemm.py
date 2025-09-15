import numpy as np
from scipy import linalg
import os
import time

#import psutil



def actual(A,B):
    #p = psutil.Process(os.getpid())
    #print("Number of threads used:", p.num_threads())
    ret = A@B
    return ret


def linalg_blas(A,B):
    ret = linalg.blas.sgemm(1.0,A,B,trans_a=False,trans_b=False)
    return ret

if __name__ == "__main__":
    
    
    #Problem size
    N = int(os.environ.get("LAAB_N", 3000))
    REPS = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = np.float32

    A = np.random.randn(N,N).astype(DTYPE)
    A = A.ravel(order='F').reshape(A.shape, order='F')
    B = np.random.randn(N,N).astype(DTYPE)
    B = B.ravel(order='F').reshape(B.shape, order='F')


    for i in range(REPS):
        start = time.perf_counter()
        ret = actual(A,B)
        end = time.perf_counter()
        elapsed_actual = end-start 
        
        start = time.perf_counter()
        ret = linalg_blas(A,B)
        end = time.perf_counter()
        elapsed_blas = end-start
        
        print("[LAAB] SciPy | sgemm | actual={:.5f} s | linalg_blas={:.5f} s".format(elapsed_actual, elapsed_blas)) 