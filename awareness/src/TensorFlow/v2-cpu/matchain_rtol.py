import tensorflow as tf
import os
import time

@tf.function
def optimized(H,x):
    ret = tf.transpose(H)@(H@x) 
    return ret

@tf.function
def actual(H,x):
    ret = tf.transpose(H)@H@x 
    return ret


if __name__ == "__main__":
    
    #Set threads
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    #Problem size
    n = int(os.environ.get("LAAB_N", 3000))
    reps = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = tf.float32

    H = tf.random.normal([n, n], dtype=DTYPE)
    x = tf.random.normal([n, 1], dtype=DTYPE)



    for i in range(reps):
        start = time.perf_counter()
        ret = actual(H,x)
        end = time.perf_counter()
        
        elapsed_actual = end-start
        
        start = time.perf_counter()
        ret = optimized(H,x)
        end = time.perf_counter()
        
        elapsed_optimized = end-start
        
        print("[LAAB] TensorFlow | matchain_rtol | optimized={:.5f} s | actual={:.5f} s".format(elapsed_optimized,elapsed_actual))  
