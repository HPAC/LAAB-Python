import tensorflow as tf
import os
import time
import numpy as np
from scipy import linalg


def optimized(A):
    ret = linalg.blas.ssyrk(1.0,A)
    #ret = A@A.T
    return ret

@tf.function
def actual(A):
    ret = A@tf.transpose(A)
    return ret

@tf.function
def linalg_matmul(A):
    ret = tf.linalg.matmul(A,tf.transpose(A))
    return ret


if __name__ == "__main__":

    #Set threads
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    #Problem size
    N = int(os.environ.get("LAAB_N", 3000))
    REPS = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = tf.float32

    A = tf.random.normal([N, N], dtype=DTYPE)
   
    DTYPE = np.float32
    A_opt = np.random.randn(N,N).astype(DTYPE)
    A_opt = A_opt.ravel(order='F').reshape(A_opt.shape, order='F')


    for i in range(REPS):
        start = time.perf_counter()
        ret = actual(A)
        end = time.perf_counter()
        elapsed_actual = end-start
        
        start = time.perf_counter()
        ret = linalg_matmul(A)
        end = time.perf_counter()
        elapsed_matmul = end-start
        
        start = time.perf_counter()
        ret = optimized(A_opt)
        end = time.perf_counter()
        elapsed_optimized = end-start 
        
        print("[LAAB] TensorFlow | mp_syrk | optimized={:.5f} s | actual={:.5f} s | linalg_matmul={:.5f} s".format(elapsed_optimized, elapsed_actual, elapsed_matmul))  
    

