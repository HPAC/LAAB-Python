import tensorflow as tf
import os
import time

@tf.function
def operator(A,B,V,ret):
    for i in range(3):
        ret = A@B + tf.tensordot(V[i],tf.transpose(V[i]),axes=0)
    return ret

@tf.function
def ref_positive(A,B,V,ret):
    tmp = A@B
    for i in range(3):
        ret = tmp + tf.tensordot(V[i],tf.transpose(V[i]),axes=0)
    return ret

if __name__ == "__main__":
    
    exp_name = os.path.basename(__file__).split(".")[0]
    
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
    V = tf.random.normal([3, n], dtype=DTYPE)
    ret = tf.random.normal([n, n], dtype=DTYPE)

    for i in range(reps):
        #cache scrub 300MB
        _ = bytearray(300*1024*1024); _[:] = b'0'
        
        start = time.perf_counter()
        ret1 = operator(A,B,V,ret)
        end = time.perf_counter()
        elapsed_operator = end-start
        
        start = time.perf_counter()
        ret1 = ref_positive(A,B,V,ret)
        end = time.perf_counter()
        elapsed_ref_positive = end-start
        elapsed_ref_negative = 3*elapsed_ref_positive

        print("[LAAB] TensorFlow | {} | ref_positive={:.5f} s | operator={:.5f} s | ref_negative={:.5f} s".format(exp_name, elapsed_ref_positive,
                                                                                                                        elapsed_operator,
                                                                                                                        elapsed_ref_negative))
