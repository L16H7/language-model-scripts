# merge_lora.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

base_model = "path/to/base"
lora_path = "path/to/lora"

# (A) sanity check: ensure the LoRA targets this base
pcfg = PeftConfig.from_pretrained(lora_path)
print("LoRA expects base:", pcfg.base_model_name_or_path)

# (B) load base in a *merge-friendly dtype* (fp16/bf16). CPU or single GPU is fine.
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype="auto",
    device_map="auto",  # or "cpu"
)

# (C) attach LoRA, then merge it into the base weights
model = PeftModel.from_pretrained(model, lora_path)
model = model.merge_and_unload()  # <- does the actual merge

# (D) save as a plain HF model folder
out_dir = "path/to/out"
model.save_pretrained(out_dir, safe_serialization=True)
tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tok.save_pretrained(out_dir)

print("Merged model saved to:", out_dir)
