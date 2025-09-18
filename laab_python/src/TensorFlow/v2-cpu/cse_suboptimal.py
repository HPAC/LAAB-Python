import tensorflow as tf
import os
import time

@tf.function
def operator(A,B,y):
    ret = tf.transpose(A)@B@tf.transpose(A)@B@y    
    return ret

@tf.function
def ref_positive(A,B,y):
    ret = tf.transpose(A)@(B@(tf.transpose(A)@(B@y)))
    return ret

@tf.function
def ref_negative(A,B,y):
    tmp1 = tf.transpose(A)@B
    tmp2  = tmp1@tmp1
    ret = tmp2@y
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
    y = tf.random.normal([n, 1], dtype=DTYPE)


    for i in range(reps):
        #cache scrub 300MB
        _ = bytearray(300*1024*1024); _[:] = b'0'
        
        start = time.perf_counter()
        ret = operator(A,B,y)
        end = time.perf_counter()
        elapsed_operator = end-start

        start = time.perf_counter()
        ret = ref_positive(A,B,y)
        end = time.perf_counter()
        elapsed_ref_positive = end-start
        
        start = time.perf_counter()
        ret = ref_negative(A,B,y)
        end = time.perf_counter()
        elapsed_ref_negative = end-start


        print("[LAAB] TensorFlow | cse_suboptimal | ref_positive={:.5f} s | operator={:.5f} s | ref_negative={:.5f} s".format(elapsed_ref_positive, 
                                                                                                                              elapsed_operator,
                                                                                                                              elapsed_ref_negative))  
