import torch
import os
import time

#(A'B)'(A'B)

@torch.jit.script
def graph_mode(A,B):
    ret = torch.t(torch.t(A)@B)@(torch.t(A)@B)    
    return ret

def eager_mode(A,B):
    ret = torch.t(torch.t(A)@B)@(torch.t(A)@B)    
    return ret

if __name__ == "__main__":

    #Sets the number of threads used for intraop parallelism on CPU.
    torch.set_num_threads(1)

    #Problem size
    n = int(os.environ.get("LAAB_N", 3000))
    reps = int(os.environ.get("LAAB_REPS", 3))
    DTYPE = torch.float32
    
    A = torch.randn([n, n], dtype=DTYPE)
    B = torch.randn([n, n], dtype=DTYPE)


    for i in range(reps):
        start = time.perf_counter()
        ret = eager_mode(A,B)
        end = time.perf_counter()
        elapsed_eager = end-start

        start = time.perf_counter()
        ret = graph_mode(A,B)
        end = time.perf_counter()
        elapsed_graph = end-start


        print("[LAAB] PyTorch | eager_vs_graph | eager={:.5f} s | graph={:.5f} s".format(elapsed_eager, elapsed_graph))  