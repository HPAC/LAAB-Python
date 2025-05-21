
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
        
        
    # Compute FLOPS and memory traffic (in bytes)
    flops = 17000000000  # 17 GFLOPS
    input_size = "3x224x224"
    load_bytes = 3*224*224*4  # float32
    store_bytes = 0
    in_intensity = flops / load_bytes if load_bytes > 0 else 0
    out_intensity = flops / store_bytes if store_bytes > 0 else 0
    exp_name = "vit_b16"
    
    log_msg = (
        f"[LAAB] {exp_name} | "
        f"args=inp_size:{input_size} | FLOPS={flops} | "
        f"load={load_bytes} | store={store_bytes} | "
        f"inp_intensity={in_intensity:.3f} | out_intensity={out_intensity:.3f}\n"
    )
    
    # Write to log file
    log_path = os.path.join(log_dir, f"{exp_name}_info.log")
    with open(log_path, "a") as f:
        f.write(log_msg)
        
if __name__ == "__main__":
    main()
