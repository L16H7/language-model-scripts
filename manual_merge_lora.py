"""
====================
Merges Unsloth LoRA checkpoint into Gemma 4 E4B base model
WITHOUT using PEFT — bypasses Gemma4ClippableLinear error completely.

  pip install torch transformers safetensors accelerate sentencepiece

Math:
  W_merged = W_base + (alpha / r) * B @ A

Usage:
  python manual_merge_mac.py \
      --base_model  google/gemma-4-e4b-it \
      --adapter_path ./gemma4-e4b-lora-checkpoints/checkpoint-200 \
      --output_dir  ./gemma4-e4b-merged
"""

import os
import json
import argparse
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoProcessor


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",   default="unsloth/gemma-4-e4b-it")
    p.add_argument("--adapter_path", required=True,
                   help="Path to your Unsloth checkpoint directory")
    p.add_argument("--output_dir",   default="./gemma4-e4b-merged")
    p.add_argument("--device",       default="cpu", choices=["cpu", "mps"])
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def load_adapter_config(adapter_path: str) -> dict:
    config_file = Path(adapter_path) / "adapter_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")
    with open(config_file) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
def load_adapter_weights(adapter_path: str) -> dict:
    """
    Loads adapter weights from safetensors or pytorch_model.bin.
    Returns dict of { key: tensor }
    """
    p = Path(adapter_path)

    # Try safetensors first (Unsloth default)
    sf_file = p / "adapter_model.safetensors"
    if sf_file.exists():
        print(f"   Loading adapter from: adapter_model.safetensors")
        return load_file(str(sf_file), device="cpu")

    # Fallback to .bin
    bin_file = p / "adapter_model.bin"
    if bin_file.exists():
        print(f"   Loading adapter from: adapter_model.bin")
        return torch.load(str(bin_file), map_location="cpu", weights_only=True)

    raise FileNotFoundError(
        f"No adapter weights found in {adapter_path}.\n"
        f"Expected: adapter_model.safetensors OR adapter_model.bin"
    )


# ─────────────────────────────────────────────────────────────────────────────
def adapter_key_to_base_key(adapter_key: str) -> str:
    """
    Maps Unsloth/PEFT adapter key → base model state_dict key.

    Unsloth format:
      base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
      base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight  (rare)

    Base model format:
      model.layers.0.self_attn.q_proj.weight
    """
    key = adapter_key
    # Strip leading "base_model.model."
    if key.startswith("base_model.model."):
        key = key[len("base_model.model."):]
    # Remove ".lora_A.weight" or ".lora_B.weight"
    key = key.replace(".lora_A.default.weight", ".weight")
    key = key.replace(".lora_B.default.weight", ".weight")
    key = key.replace(".lora_A.weight", ".weight")
    key = key.replace(".lora_B.weight", ".weight")
    return key


# ─────────────────────────────────────────────────────────────────────────────
def merge_lora_into_state_dict(
    state_dict:      dict,
    adapter_weights: dict,
    scaling:         float,
    verbose:         bool = True,
) -> tuple[dict, int, list]:
    """
    Applies ΔW = scaling * B @ A into the base model state_dict in-place.
    Returns (modified_state_dict, num_merged, skipped_keys).
    """

    # Collect all lora_A keys
    lora_A_keys = [k for k in adapter_weights if "lora_A" in k]

    merged_count = 0
    skipped      = []

    for key_A in sorted(lora_A_keys):
        key_B = key_A.replace("lora_A", "lora_B")

        if key_B not in adapter_weights:
            print(f"   ⚠️  No matching lora_B for {key_A} — skipping")
            skipped.append(key_A)
            continue

        base_key = adapter_key_to_base_key(key_A)

        if base_key not in state_dict:
            # Try stripping one more "model." prefix (happens in some MoE configs)
            alt_key = base_key.replace("model.", "", 1)
            if alt_key in state_dict:
                base_key = alt_key
            else:
                if verbose:
                    print(f"   ⚠️  Base key not found: {base_key} — skipping")
                skipped.append(base_key)
                continue

        A = adapter_weights[key_A].float()   # [r, in_features]
        B = adapter_weights[key_B].float()   # [out_features, r]

        delta_W = scaling * (B @ A)          # [out_features, in_features]

        # Cast delta to match base weight dtype
        base_dtype = state_dict[base_key].dtype
        state_dict[base_key] = (
            state_dict[base_key].float() + delta_W
        ).to(base_dtype)

        if verbose:
            print(f"   ✅ merged: {base_key:70s} Δ={delta_W.abs().mean():.6f}")

        merged_count += 1

    return state_dict, merged_count, skipped


# ─────────────────────────────────────────────────────────────────────────────
def load_base_model(base_model: str, device: str):
    """
    Loads Gemma 4 E4B with the correct transformers class.
    Requires transformers >= 4.51.
    """
    try:
        from transformers import Gemma4ForConditionalGeneration
        print("   Using: Gemma4ForConditionalGeneration")
        ModelClass = Gemma4ForConditionalGeneration
    except ImportError:
        print("   ⚠️  Gemma4ForConditionalGeneration not available.")
        print("   Run: pip install --upgrade transformers")
        print("   Falling back to AutoModel (may fail for multimodal)")
        from transformers import AutoModel
        ModelClass = AutoModel

    model = ModelClass.from_pretrained(
        base_model,
        torch_dtype    = torch.bfloat16,
        device_map     = device,
        trust_remote_code = True,
        low_cpu_mem_usage = True,
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print("\n" + "═" * 65)
    print("  Gemma 4 E4B — Manual LoRA Merge (no PEFT, Mac-compatible)")
    print("═" * 65)
    print(f"  Base model   : {args.base_model}")
    print(f"  Adapter path : {args.adapter_path}")
    print(f"  Output dir   : {args.output_dir}")
    print(f"  Device       : {args.device}")
    print("═" * 65 + "\n")

    # ── 1. Read adapter config ─────────────────────────────────────
    print("📋 Reading adapter config ...")
    config  = load_adapter_config(args.adapter_path)
    r       = config["r"]
    alpha   = config.get("lora_alpha", r)
    scaling = alpha / r
    print(f"   r          = {r}")
    print(f"   lora_alpha = {alpha}")
    print(f"   scaling    = alpha/r = {scaling:.4f}")
    print(f"   target_modules: {config.get('target_modules', 'N/A')}\n")

    # ── 2. Load adapter weights ────────────────────────────────────
    print("🔌 Loading adapter weights ...")
    adapter_weights = load_adapter_weights(args.adapter_path)
    lora_keys = [k for k in adapter_weights if "lora_A" in k]
    print(f"   Found {len(lora_keys)} LoRA A/B layer pairs\n")

    # ── 3. Load base model ─────────────────────────────────────────
    print(f"🧠 Loading base model onto {args.device} ...")
    print("   E4B in bfloat16 ≈ 8 GB RAM — please wait ...\n")
    model = load_base_model(args.base_model, args.device)
    print("   ✅ Base model loaded\n")

    # ── 4. Get state dict ──────────────────────────────────────────
    print("📦 Extracting state dict ...")
    state_dict = model.state_dict()
    print(f"   Base model has {len(state_dict)} weight tensors\n")

    # Debug: show adapter key → base key mapping for first 3
    print("🔍 Key mapping preview (first 3 pairs):")
    for k in sorted(lora_keys)[:3]:
        mapped = adapter_key_to_base_key(k)
        found  = "✅" if mapped in state_dict else "❌ NOT FOUND"
        print(f"   {k}")
        print(f"   → {mapped}  {found}\n")

    # ── 5. Merge ───────────────────────────────────────────────────
    print("🔀 Merging LoRA weights (W = W₀ + α/r · B·A) ...")
    print("─" * 65)
    state_dict, num_merged, skipped = merge_lora_into_state_dict(
        state_dict, adapter_weights, scaling, verbose=True
    )
    print("─" * 65)
    print(f"\n   ✅ Merged:  {num_merged} layers")
    if skipped:
        print(f"   ⚠️  Skipped: {len(skipped)} keys")
        for s in skipped:
            print(f"      - {s}")
    print()

    # ── 6. Load merged weights back into model ─────────────────────
    print("📥 Loading merged weights back into model ...")
    model.load_state_dict(state_dict, strict=True)
    print("   ✅ Done\n")

    # ── 7. Save ───────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"💾 Saving merged model to: {args.output_dir}")

    model.save_pretrained(
        args.output_dir,
        safe_serialization = True,
        max_shard_size     = "4GB",
    )

    # Save tokenizer + processor
    try:
        processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
        processor.save_pretrained(args.output_dir)
        print("   ✅ Processor saved")
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)
    print("   ✅ Tokenizer saved")

    # ── 8. Summary ─────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  ✅ MERGE COMPLETE")
    print("═" * 65)
    print(f"  Output: {args.output_dir}")
    print("\n  Files saved:")
    for f in sorted(os.listdir(args.output_dir)):
        fpath = Path(args.output_dir) / f
        size  = fpath.stat().st_size / (1024**3)
        bar   = f"{size:.2f} GB" if size > 0.01 else "   <1 MB"
        print(f"    {f:50s} {bar}")

    print("\n  Next → convert to GGUF:")
    print("    bash convert_gguf.sh\n")


if __name__ == "__main__":
    main()