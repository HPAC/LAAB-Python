from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load a small open model (can scale this up)
model_name = "gpt2"  # or try "EleutherAI/gpt-neo-125M", "tiiuae/falcon-rw-1b"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare fake input (e.g., "hello world")
dummy_input = tokenizer("hello world", return_tensors="pt")

# One forward pass (no gradients)
with torch.no_grad():
    output = model(**dummy_input)

# Output contains logits: [batch, seq_len, vocab_size]
print("Output shape:", output.logits.shape)

