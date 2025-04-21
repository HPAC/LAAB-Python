import torch
import os
import time


#Sets the number of threads used for intraop parallelism on CPU.
torch.set_num_threads(1)

#Problem size
n = 3000
reps = 3
DTYPE = torch.float32


@torch.jit.script
def actual_expr(A,B):
    ret = torch.t(A)@B + torch.t(A)@B
    return ret

@torch.jit.script
def simplified_expr(A,B):
    tmp = torch.t(A)@B
    ret = tmp + tmp
    return ret


A = torch.randn([n, n], dtype=DTYPE)
B = torch.randn([n, n], dtype=DTYPE)


for i in range(reps):
   start = time.perf_counter()
   ret1 = actual_expr(A,B)
   end = time.perf_counter()
   elapsed = end-start

   start = time.perf_counter()
   ret1 = simplified_expr(A,B)
   end = time.perf_counter()
   optimized = end-start


   print("PyTorch | cse_addition | elapsed={:.5f} s | optimized={:.5f} s".format(elapsed,optimized))  
