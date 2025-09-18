import tensorflow as tf
import os
import time


@tf.function
def ref_positive(A,H,x):
    ret = A@x - tf.transpose(H)@(H@x)
    return ret

@tf.function
def operator(A,H,x):
    ret = (A - tf.transpose(H)@H)@x
    return ret

@tf.function
def ref_negative(A,H,x):
    tmp = tf.transpose(H)@H
    ret = (A - tmp)@x
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
        #cache scrub 300MB
        _ = bytearray(300*1024*1024); _[:] = b'0'
        
        start = time.perf_counter()
        ret1 = operator(A,H,x)
        end = time.perf_counter()
        elapsed_operator = end-start

        start = time.perf_counter()
        ret1 = ref_positive(A,H,x)
        end = time.perf_counter()
        elapsed_ref_positive = end-start
        
        start = time.perf_counter()
        ret1 = ref_negative(A,H,x)
        end = time.perf_counter()
        elapsed_ref_negative = end-start    
        
        print("[LAAB] TensorFlow | am_distributivity2 | ref_positive={:.5f} s | operator={:.5f} s | ref_negative={:.5f} s".format(elapsed_ref_positive,
                                                                                                                                  elapsed_operator,
                                                                                                                                  elapsed_ref_negative))

