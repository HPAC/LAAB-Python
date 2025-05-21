import torch
import sys
from transformers import ViTModel

# # --- Bug in Pytorch 2.1.2 - Monkey-patch torch.get_default_device ---
if not hasattr(torch, "get_default_device"):
    torch.get_default_device = lambda: "cuda" if torch.cuda.is_available() else "cpu"
# # ----------------------------------------------------

if len(sys.argv) != 2 or sys.argv[1] not in ("cpu", "cuda"):
    print("Usage: python script.py [cpu|cuda]")
    sys.exit(1)

# Explicit device selection
device = torch.device(sys.argv[1])
print(f"Using device: {device}")

# Load pretrained ViT-B/16 from Hugging Face (no device_map, move manually)
model_name = "google/vit-base-patch16-224"
model = ViTModel.from_pretrained(model_name, device_map=None)
model.to(device)
model.eval()

# Fake input: [batch_size, 3, 224, 224]
batch_size = 4
fake_images = torch.randn(batch_size, 3, 224, 224).to(device)

# Number of repetitions
REP = 10

# Forward passes
for i in range(REP):
    with torch.no_grad():
        outputs = model(pixel_values=fake_images)
    print(f"Run {i+1}: CLS embedding shape = {outputs.pooler_output.shape}")

