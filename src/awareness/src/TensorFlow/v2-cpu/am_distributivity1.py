import tensorflow as tf
import os
import time


@tf.function
def actual(A,B,C):
    ret = A@B + A@C
    return ret

@tf.function
def optimized(A,B,C):
    ret = A@(B+C)
    return ret

if __name__ == "__main__":

    #Set threads
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    #Problem size
    n = int(os.environ.get("LAAB_N", 3000))
    reps = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = tf.float32

    A = tf.random.normal([n, n], dtype=DTYPE)
    B = tf.random.normal([n, n], dtype=DTYPE)
    C = tf.random.normal([n, n], dtype=DTYPE)

    for i in range(reps):
        start = time.perf_counter()
        ret1 = actual(A,B,C)
        end = time.perf_counter()
        elapsed_actual = end-start

        start = time.perf_counter()
        ret1 = optimized(A,B,C)
        end = time.perf_counter()
        elapsed_optimized = end-start    
        
        print("[LAAB] TensorFlow | am_distributivity1 | optimized={:.5f} s | actual={:.5f} s".format(elapsed_optimized,elapsed_actual))

