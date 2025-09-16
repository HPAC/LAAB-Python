import tensorflow as tf
import os
import time


@tf.function
def optimized(A,H,x):
    ret = A@x - tf.transpose(H)@(H@x)
    return ret

@tf.function
def actual(A,H,x):
    ret = (A - tf.transpose(H)@H)@x
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
    H = tf.random.normal([n, n], dtype=DTYPE)
    x = tf.random.normal([n, 1], dtype=DTYPE)

    for i in range(reps):
        start = time.perf_counter()
        ret1 = actual(A,H,x)
        end = time.perf_counter()
        elapsed_actual = end-start

        start = time.perf_counter()
        ret1 = optimized(A,H,x)
        end = time.perf_counter()
        elapsed_optimized = end-start    
        
        print("[LAAB] TensorFlow | am_distributivity2 | optimized={:.5f} s | actual={:.5f} s".format(elapsed_optimized,elapsed_actual))

