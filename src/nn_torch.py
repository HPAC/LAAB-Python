import os
import time
import torch
import torch.nn as nn

class bcolors:
    WARNING = '\033[93m'
    ENDC = '\033[0m'

# Define GEMM as a Linear layer wrapped in nn.Sequential
class GEMMModel(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(N, N, bias=True)
        )

    def forward(self, x):
        return self.seq(x)

if __name__ == "__main__":
    # torch.backends.mkl.is_available() check (optional)
    # print(bcolors.WARNING + "MKL Enabled : ", torch.backends.mkl.is_available(), bcolors.ENDC)

    torch.set_num_threads(1)

    # Problem size
    N = int(os.environ["LAMP_N"]) if "LAMP_N" in os.environ else 3000
    REPS = int(os.environ["LAMP_REPS"]) if "LAMP_REPS" in os.environ else 1
    DTYPE = torch.float32

    # Input and model
    A = torch.randn(N, N, dtype=DTYPE)
    model = GEMMModel(N).to(dtype=DTYPE)
    # Initialize weights like B
    with torch.no_grad():
        model.seq[0].weight.copy_(torch.randn(N, N, dtype=DTYPE))

    # Warm-up
    ret = model(A)

    # Timed reps
    for i in range(REPS):
        start = time.perf_counter()
        ret = model(A)
        end = time.perf_counter()
        print(f"gemm_torch : {end - start:.6f} seconds")

