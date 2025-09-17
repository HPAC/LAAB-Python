import tensorflow as tf
import os
import time


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
    THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))
    tf.config.threading.set_inter_op_parallelism_threads(THREADS)
    tf.config.threading.set_intra_op_parallelism_threads(THREADS)

    #Problem size
    N = int(os.environ.get("LAAB_N", 3000))
    REPS = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = tf.float32

    A = tf.random.normal([N, N], dtype=DTYPE)


    for i in range(REPS):
        #cache scrub 300MB
        _ = bytearray(300*1024*1024); _[:] = b'0'
        
        start = time.perf_counter()
        ret = actual(A)
        end = time.perf_counter()
        elapsed_actual = end-start
        
        start = time.perf_counter()
        ret = linalg_matmul(A)
        end = time.perf_counter()
        elapsed_matmul = end-start
        
        print("[LAAB] TensorFlow | mp_syrk | actual={:.5f} s | linalg_matmul={:.5f} s".format(elapsed_actual, elapsed_matmul))  
    

