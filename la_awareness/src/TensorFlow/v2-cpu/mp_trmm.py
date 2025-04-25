import tensorflow as tf
import os
import time
import numpy as np
from scipy import linalg


@tf.function
def actual(A,B):
    ret = A@B
    return ret

@tf.function
def linalg_matmul(A,B):
    ret = tf.linalg.matmul(A,B)
    return ret

def optimized(A,B):
    ret = linalg.blas.strmm(1.0, A, B, diag=False, trans_a=False, side=False,lower=True)  # diag specifies unit triangular
    return ret

if __name__ == "__main__":

    #Set threads
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    #Problem size
    N = int(os.environ.get("LAAB_N", 3000))
    REPS = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = tf.float32

    A = tf.linalg.band_part(tf.random.normal([N, N], dtype=DTYPE),-1,0)
    B = tf.random.normal([N, N], dtype=DTYPE)
   
    DTYPE_OPT = np.float32 
    A_opt = np.tril(np.random.randn(N,N).astype(DTYPE_OPT))
    A_opt = A_opt.ravel(order='F').reshape(A_opt.shape, order='F')
    B_opt = np.random.randn(N,N).astype(DTYPE_OPT)
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
        
        print("[LAAB] TensorFlow | mp_trmm | optimized={:.5f} s | actual={:.5f} s | linalg_matmul={:.5f} s".format(elapsed_optimized, elapsed_actual, elapsed_matmul))  
    

