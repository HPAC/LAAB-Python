import tensorflow as tf
import os
import time


@tf.function
def actual(A,B):
    ret = tf.transpose(tf.transpose(A)@B)@(tf.transpose(A)@B)    
    return ret

@tf.function
def optimized(A,B):
    tmp = tf.transpose(A)@B
    ret = tf.transpose(tmp)@tmp
    return ret


if __name__ == "__main__":

    #Set threads
    THREADS = int(os.environ.get("OMP_NUM_THREADS", 1))
    tf.config.threading.set_inter_op_parallelism_threads(THREADS)
    tf.config.threading.set_intra_op_parallelism_threads(THREADS)

    #Problem size
    n = int(os.environ.get("LAAB_N", 3000))
    reps = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = tf.float32
    
    A = tf.random.normal([n, n], dtype=DTYPE)
    B = tf.random.normal([n, n], dtype=DTYPE)


    for i in range(reps):
        start = time.perf_counter()
        ret = actual(A,B)
        end = time.perf_counter()
        elapsed_actual = end-start

        start = time.perf_counter()
        ret = optimized(A,B)
        end = time.perf_counter()
        elapsed_optimized = end-start


        print("[LAAB] TensorFlow | cse_matmul_paranthesis | optimized={:.5f} s | actual={:.5f} s".format(elapsed_optimized, elapsed_actual))  
