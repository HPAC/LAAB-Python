import tensorflow as tf
import os
import time

@tf.function
def actual(A,B,V,ret):
    for i in range(3):
        ret = A@B + tf.tensordot(V[i],tf.transpose(V[i]),axes=0)
    return ret

@tf.function
def optimized(A,B,V,ret):
    tmp = A@B
    for i in range(3):
        ret = tmp + tf.tensordot(V[i],tf.transpose(V[i]),axes=0)
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
    V = tf.random.normal([3, n], dtype=DTYPE)
    ret = tf.random.normal([n, n], dtype=DTYPE)

    for i in range(reps):
        #cache scrub 300MB
        _ = bytearray(300*1024*1024); _[:] = b'0'
        
        start = time.perf_counter()
        ret1 = actual(A,B,V,ret)
        end = time.perf_counter()
        elapsed_actual = end-start
        
        start = time.perf_counter()
        ret1 = optimized(A,B,V,ret)
        end = time.perf_counter()
        elapsed_optimized = end-start

        print("[LAAB] TensorFlow | cm_loops | optimized={:.5f} s | actual={:.5f} s".format(elapsed_optimized,elapsed_actual))
