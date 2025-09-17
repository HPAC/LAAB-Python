import tensorflow as tf
import os
import time


@tf.function
def actual(A,B):
    ret = A@B
    return ret

@tf.function
def linalg_matmul(A,B):
    ret = tf.linalg.matmul(A,B)
    return ret

if __name__ == "__main__":

    THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))
    tf.config.threading.set_inter_op_parallelism_threads(THREADS)
    tf.config.threading.set_intra_op_parallelism_threads(THREADS)

    #Problem size
    N = int(os.environ.get("LAAB_N", 3000))
    REPS = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = tf.float32

    A = tf.linalg.band_part(tf.random.normal([N, N], dtype=DTYPE),-1,0)
    B = tf.random.normal([N, N], dtype=DTYPE)


    for i in range(REPS):
        #cache scrub 300MB
        _ = bytearray(300*1024*1024); _[:] = b'0'
        
        start = time.perf_counter()
        ret = actual(A,B)
        end = time.perf_counter()
        elapsed_actual = end-start
        
        start = time.perf_counter()
        ret = linalg_matmul(A,B)
        end = time.perf_counter()
        elapsed_matmul = end-start
        
        print("[LAAB] TensorFlow | mp_trmm | actual={:.5f} s | linalg_matmul={:.5f} s".format(elapsed_actual, elapsed_matmul))  
    

