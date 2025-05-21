import torch
import sys
from transformers import ViTModel, ViTConfig
import socket
from laab_utils.cuda_device_info import get_cuda_pci_bus_id
import time
import os
import psutil

# # --- Bug in Pytorch 2.1.2 - Monkey-patch torch.get_default_device ---
if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: "cuda" if torch.cuda.is_available() else "cpu"
# # ----------------------------------------------------

def main():
    # Ensure device argument is passed
    if len(sys.argv) != 3 or sys.argv[1] not in ("cpu", "cuda") or not sys.argv[2].isdigit():
        print("Usage: python script.py [cpu|cuda] [batch_size:int]")
        sys.exit(1)

    device = torch.device(sys.argv[1])
    batch_size = int(sys.argv[2])
    if batch_size <= 0:
        print("Batch size must be a positive integer.")
        sys.exit(1)    
    print(f"Using device: {device}")
    print(f"Batch size: {batch_size}")
    
    REP = int(os.getenv("LAAB_REPS", "10"))
    
    log_dir = os.getenv("LAAB_LOG_DIR")
    if not log_dir:
        print("Environment variable LAAB_LOG_DIR not set.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(log_dir, exist_ok=True)

    hostname = socket.gethostname()
    log_path = os.path.join(log_dir, f"vit_b16.{hostname}.log")
    log_file = open(log_path, "a")
    pci_bus_id = -1
    if device.type == "cuda":
        pci_bus_id = get_cuda_pci_bus_id(torch.cuda.current_device())

    try:
        cpu = psutil.Process().cpu_num()
    except AttributeError:
        cpu = -1  # fallback for unsupported platforms
        
    # Load pretrained ViT-B/16
    model_name = "google/vit-base-patch16-224"
    model = ViTModel.from_pretrained(model_name, device_map=None)
    model.to(device)
    model.eval()
    

    # Fake input
    gflops = batch_size*17
    
    # Run REP times
    for i in range(REP):
        fake_images = torch.randn(batch_size, 3, 224, 224)
        
        # Start timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        if device.type == "cuda":
            fake_images = fake_images.to(device)
            
        with torch.no_grad():
            outputs = model(pixel_values=fake_images)
            
        # End timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        
        seconds = end - start
        gflops_per_sec = gflops / seconds
        
        now = time.localtime()
        datetime = time.strftime("%Y-%m-%d %H:%M:%S", now)
        
        if device.type == "cuda":
            log_file.write(
                f"[LAAB] vit_b16_cuda | datetime={datetime} | batch={batch_size} | duration={seconds:.3f} s | perf={gflops_per_sec:.2f} GFLOP/s | CPU={cpu} | BUS={pci_bus_id}\n"
            )
        else:
            log_file.write(
                f"[LAAB] vit_b16_cpu | datetime={datetime} | batch={batch_size} | duration={seconds:.3f} s | perf={gflops_per_sec:.2f} GFLOP/s | CPU={cpu}\n"
            )
        log_file.flush()
                
    log_file.close()
    
if __name__ == "__main__":
    main()

