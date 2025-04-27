import numpy as np
import os
import time

class bcolors:
    WARNING = '\033[93m'
    ENDC = '\033[0m'


def gemm_np(A,B):
    ret = A@B
    return ret


if __name__ == "__main__":

    #Check if MKL is enabled
    #import tensorflow.python.framework as tff
    #print(bcolors.WARNING + "MKL Enabled : ", tff.test_util.IsMklEnabled(), bcolors.ENDC)

    #Set threads
    #tf.config.threading.set_inter_op_parallelism_threads(1)
    #tf.config.threading.set_intra_op_parallelism_threads(1)

    #Problem size
    N = os.environ["LAMP_N"] if "LAMP_N" in os.environ else 3000
    REPS = os.environ["LAMP_REPS"] if "LAMP_N" in os.environ else 1
    DTYPE = np.float32


    A = np.random.normal(size=[N, N]).astype(DTYPE)
    B = np.random.normal(size=[N, N]).astype(DTYPE)


    for i in range(REPS):
        start = time.perf_counter()
        ret = gemm_np(A,B)
        end = time.perf_counter()
        print("gemm np : ", end-start)

