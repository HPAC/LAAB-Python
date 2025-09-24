import tensorflow as tf
import os
import time

@tf.function
def ref_positive(H,x,y):
    ret = (tf.transpose(H)@y)@(tf.transpose(x)@H) 
    return ret

@tf.function
def ref_negative(H,x,y):
    # evaluates from right-to-left
    tmp1 = tf.transpose(H)@y
    tmp2 = tmp1@tf.transpose(x)
    ret = tmp2@H
    return ret

@tf.function
def operator(H,x,y):
    ret = tf.transpose(H)@y@tf.transpose(x)@H 
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
    
    H = tf.random.normal([n, n], dtype=DTYPE)
    x = tf.random.normal([n, 1], dtype=DTYPE)
    y = tf.random.normal([n, 1], dtype=DTYPE)



    for i in range(reps):
        #cache scrub 300MB
        _ = bytearray(300*1024*1024); _[:] = b'0'
        
        start = time.perf_counter()
        ret = operator(H,x,y)
        end = time.perf_counter()
        
        elapsed_operator = end-start
        
        start = time.perf_counter()
        ret = ref_positive(H,x,y)
        end = time.perf_counter()
        
        elapsed_ref_positive = end-start
        
        start = time.perf_counter()
        ret = ref_negative(H,x,y)
        end = time.perf_counter()
        
        elapsed_ref_negative = end-start
        
        print("[LAAB] TensorFlow | {} | ref_positive={:.5f} s | operator={:.5f} s | ref_negative={:.5f} s".format(exp_name, elapsed_ref_positive,
                                                                                                                              elapsed_operator,
                                                                                                                              elapsed_ref_negative))  
