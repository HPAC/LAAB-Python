import os
os.environ["OMP_NUM_THREADS"] = "1"
import time
import numpy as np
from scipy import linalg

class bcolors:
    WARNING = '\033[93m'
    ENDC = '\033[0m'


def gemm_scipy(A,B):
    ret = linalg.blas.sgemm(1.0,A,B,trans_a=False,trans_b=False)
    #ret = A@B
    return ret



if __name__ == "__main__":


    #Problem size
    N = os.environ["LAMP_N"] if "LAMP_N" in os.environ else 3000
    REPS = os.environ["LAMP_REPS"] if "LAMP_N" in os.environ else 1
    DTYPE = np.float32

    A = np.random.randn(N,N).astype(DTYPE)
    A = A.ravel(order='F').reshape(A.shape, order='F')
    B = np.random.randn(N,N).astype(DTYPE)
    B = B.ravel(order='F').reshape(B.shape, order='F')

    for i in range(REPS):
        start = time.perf_counter()
        ret = gemm_scipy(A,B)
        end = time.perf_counter()
        print("gemm_scipy: ", end-start)

