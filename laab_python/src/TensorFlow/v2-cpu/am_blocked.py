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

@tf.function
def optimized(A1,A2,B1,B2):
    ret = tf.concat((A1@B1, A2@B2),0)
    return ret

if __name__ == "__main__":

    #Set threads
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    #Problem size
    n = int(os.environ.get("LAAB_N", 3000))
    reps = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = tf.float32
    nb = int(n/2)

    A1 = tf.random.normal([nb, nb], dtype=DTYPE)
    A2 = tf.random.normal([nb, nb], dtype=DTYPE)
    A = tf.concat((tf.concat((A1, tf.zeros([nb,nb])) ,1),tf.concat((tf.zeros([nb,nb]),A2) ,1)),0)

    B1 = tf.random.normal([nb, n], dtype=DTYPE)
    B2 = tf.random.normal([nb, n], dtype=DTYPE)
    B = tf.concat((B1,B2),0)

    for i in range(reps):
        start = time.perf_counter()
        ret1 = actual(A,B)
        end = time.perf_counter()
        elapsed_actual = end-start
        
        start = time.perf_counter()
        ret = linalg_matmul(A,B)
        end = time.perf_counter()
        elapsed_matmul = end-start
        
        start = time.perf_counter()
        ret1 = optimized(A1,A2,B1,B2)
        end = time.perf_counter()
        elapsed_optimized = end-start    

        print("[LAAB] TensorFlow | am_blocked | optimized={:.5f} s | actual={:.5f} s  | linalg_matmul={:.5f} s".format(elapsed_optimized, elapsed_actual, elapsed_matmul))
