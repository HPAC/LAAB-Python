import os
import time
import socket
import psutil
import torch
import numpy as np

SCRUB_SIZE = 50 * 1024 * 1024  # ~200 MB
dtype = torch.float64  # Double precision

def cache_scrub():
    scrub = torch.zeros(SCRUB_SIZE, dtype=dtype)
    scrub += 1  # ensure it touches memory
    del scrub

def main():
    import sys
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} m k n")
        sys.exit(1)

    log_dir = os.getenv("LAAB_LOG_DIR")
    if not log_dir:
        print("Environment variable LAAB_LOG_DIR not set.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(log_dir, exist_ok=True)

    hostname = socket.gethostname()
    log_path = os.path.join(log_dir, f"dgemm_cpu.{hostname}.log")
    log_file = open(log_path, "a")

    try:
        cpu = psutil.Process().cpu_num()
    except AttributeError:
        cpu = -1  # fallback for unsupported platforms

    REP = int(os.getenv("LAAB_REPS", "3"))

    m, k, n = map(int, sys.argv[1:4])

    np.random.seed(int(time.time()))
    A = torch.tensor(np.random.rand(m, k), dtype=dtype)
    B = torch.tensor(np.random.rand(k, n), dtype=dtype)
    C = torch.empty((m, n), dtype=dtype)

    for it in range(REP):
        C.zero_()
        cache_scrub()

        start = time.perf_counter()
        C[:] = torch.matmul(A, B)
        end = time.perf_counter()

        elapsed = end - start
        flops = 2.0 * m * n * k
        gflops = flops / elapsed / 1e9

        if it % 50 == 0:
            now = time.localtime()
            datetime = time.strftime("%Y-%m-%d %H:%M:%S", now)
            log_file.write(
                f"[LAAB] dgemm_cpu | datetime={datetime} | duration={elapsed:.3f} s | perf={gflops:.2f} GFLOP/s | CPU={cpu}\n"
            )
            log_file.flush()

    log_file.close()

if __name__ == "__main__":
    main()
