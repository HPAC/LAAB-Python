import os
import time
import socket
import torch
import numpy as np
import psutil
import ctypes
import ctypes.util
from laab_utils.cuda_device_info import get_cuda_pci_bus_id

dtype = torch.float64  # double precision


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
    log_path = os.path.join(log_dir, f"dgemm_cuda.{hostname}.log")
    log_file = open(log_path, "a")

    try:
        cpu = psutil.Process().cpu_num()
    except AttributeError:
        cpu = -1

    REP = int(os.getenv("LAAB_REPS", "3"))

    m, k, n = map(int, sys.argv[1:4])

    # GPU device info
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.", file=sys.stderr)
        sys.exit(1)
        
    device = torch.cuda.current_device()
    pci_bus_id = get_cuda_pci_bus_id(device)
    if pci_bus_id is None:
        print("Failed to get PCI Bus ID. Exiting.", file=sys.stderr)
        sys.exit(1)
    #visible_dev = os.getenv("CUDA_VISIBLE_DEVICES", "not set")

    #print(f"[INFO] Running on CUDA device: {device.index} (CUDA_VISIBLE_DEVICES={visible_dev})")

    # Random input
    torch.manual_seed(int(time.time()))
    A = torch.rand((m, k), dtype=dtype, device=device)
    B = torch.rand((k, n), dtype=dtype, device=device)
    C = torch.empty((m, n), dtype=dtype, device=device)

    for it in range(REP):
        C.zero_()

        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start.record()

        C[:] = torch.matmul(A, B)

        stop.record()
        torch.cuda.synchronize()

        milliseconds = start.elapsed_time(stop)
        seconds = milliseconds / 1000.0

        gflops = (2.0 * m * n * k) / 1e9
        gflops_per_sec = gflops / seconds

        if it % 50 == 0:
            now = time.localtime()
            datetime = time.strftime("%Y-%m-%d %H:%M:%S", now)
            log_file.write(
                f"[LAAB] dgemm_cuda | datetime={datetime} | duration={seconds:.3f} s | perf={gflops_per_sec:.2f} GFLOP/s | GPU={device} | BUS={pci_bus_id}\n"
            )
            log_file.flush()

    log_file.close()

if __name__ == "__main__":
    main()
