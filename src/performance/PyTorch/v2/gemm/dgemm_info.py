
import os
import sys

def main():
    # Get LAAB_LOG_DIR environment variable
    log_dir = os.getenv("LAAB_LOG_DIR")
    if not log_dir:
        print("Environment variable LAAB_LOG_DIR not set.", file=sys.stderr)
        sys.exit(1)

    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Parse command-line arguments: m, n, k
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} exp_name m n k", file=sys.stderr)
        sys.exit(1)

    try:
        exp_name = sys.argv[1]
        m = int(sys.argv[2])
        k = int(sys.argv[3])
        n = int(sys.argv[4])
    except ValueError:
        print("Arguments m, n, k must be integers.", file=sys.stderr)
        sys.exit(1)
        
        
    # Compute FLOPS and memory traffic (in bytes)
    flops = 2 * m * n * k
    load_bytes = (m * k + k * n) * 8  # double precision = 8 bytes
    store_bytes = m * n * 8
    in_intensity = flops / load_bytes if load_bytes > 0 else 0
    out_intensity = flops / store_bytes if store_bytes > 0 else 0
    
    log_msg = (
        f"[LAAB] {exp_name} | "
        f"args=m:{m}, n:{n}, k:{k} | FLOPS={flops} | "
        f"load={load_bytes} | store={store_bytes} | "
        f"inp_intensity={in_intensity:.3f} | out_intensity={out_intensity:.3f}\n"
    )
    
    # Write to log file
    log_path = os.path.join(log_dir, "dgemm_info.log")
    with open(log_path, "a") as f:
        f.write(log_msg)
        
if __name__ == "__main__":
    main()