"""
╔══════════════════════════════════════════════════════════════════╗
║  Gemma 4 E4B — LoRA Fine-tuning (Unsloth)                      ║
║  • Multiturn conversational dataset (local JSONL)               ║
║  • Train on ALL tokens                                          ║
║  • Context length: 10,000 tokens                                ║
║  • Saves: merged full weights + GGUF (q4_k_m, q8_0, f16)       ║
║  • Target hardware: 1× H100 80GB (works on 24GB+ with 4-bit)   ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
  pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
  python prepare_data.py
  python sanity_check.py
  python gemma4_e4b_lora.py
"""

import os
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ═══════════════════════════════════════════════════════════════
#  1. CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Model
MODEL_NAME       = "google/gemma-4-E4b-it"
MAX_SEQ_LENGTH   = 10_000
HF_TOKEN         = None                       # set if gated: "hf_..."

# Dataset
DATASET_PATH     = "./train_data.jsonl"       # output from prepare_data.py
MESSAGES_FIELD   = "messages"

# LoRA config
LORA_R           = 64          # rank — higher = more capacity, more VRAM
LORA_ALPHA       = 64          # alpha == r is a safe default
LORA_DROPOUT     = 0           # Unsloth optimized for 0
TARGET_MODULES   = [           # all linear layers
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Quantization — pick ONE
LOAD_IN_4BIT     = False       # QLoRA (fits on 24GB GPUs)
LOAD_IN_16BIT    = True        # 16-bit LoRA (better quality, needs more VRAM)

# Training hyperparameters
NUM_EPOCHS       = 3
BATCH_SIZE       = 2
GRAD_ACCUM       = 4           # effective batch = 2 * 4 = 8
LEARNING_RATE    = 1e-5        # LoRA can handle higher LR than FFT
WARMUP_STEPS     = 10
WEIGHT_DECAY     = 0.01
LR_SCHEDULER     = "cosine"
MAX_GRAD_NORM    = 1.0
OPTIMIZER        = "adamw_8bit"
LOGGING_STEPS    = 1
SAVE_STEPS       = 40
SEED             = 42

# Output directories
OUTPUT_DIR       = "./gemma4-e4b-lora-checkpoints"
MERGED_SAVE_DIR  = "./gemma4-e4b-lora-merged"
GGUF_SAVE_DIR    = "./gemma4-e4b-lora-gguf"

# GGUF quantization methods to export
GGUF_QUANTS      = ["Q8_0", "F16"]


# ═══════════════════════════════════════════════════════════════
#  2. LOAD MODEL + ATTACH LoRA
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print(f"Loading {MODEL_NAME}")
print(f"  LoRA r={LORA_R}, alpha={LORA_ALPHA}")
print(f"  {'4-bit QLoRA' if LOAD_IN_4BIT else '16-bit LoRA'}")
print("=" * 60)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = MAX_SEQ_LENGTH,
    load_in_4bit    = LOAD_IN_4BIT,
    load_in_16bit   = LOAD_IN_16BIT,
    full_finetuning = False,
    token           = HF_TOKEN,
)

model = FastLanguageModel.get_peft_model(
    model,
    r                         = LORA_R,
    target_modules            = TARGET_MODULES,
    lora_alpha                = LORA_ALPHA,
    lora_dropout              = LORA_DROPOUT,
    bias                      = "none",
    use_gradient_checkpointing = "unsloth",   # async offload — saves VRAM
    random_state              = SEED,
    max_seq_length            = MAX_SEQ_LENGTH,
)

total_params    = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params:     {total_params:,}")
print(f"Trainable params: {trainable_params:,} ({trainable_params/total_params:.2%})")


# ═══════════════════════════════════════════════════════════════
#  3. LOAD & FORMAT DATASET
# ═══════════════════════════════════════════════════════════════

print("\nLoading dataset...")
dataset = load_dataset("json", data_files={"train": DATASET_PATH}, split="train")
print(f"Dataset size: {len(dataset):,} examples")


def formatting_func(examples):
    texts = []
    for convo in examples[MESSAGES_FIELD]:
        text = tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False,
        )
        texts.append(text)
    return {"text": texts}


dataset = dataset.map(formatting_func, batched=True, remove_columns=dataset.column_names)

print("\n--- Example 0 (first 500 chars) ---")
print(dataset[0]["text"][:500])
print("--- end ---\n")


# ═══════════════════════════════════════════════════════════════
#  4. CREATE TRAINER
# ═══════════════════════════════════════════════════════════════

trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    train_dataset   = dataset,
    args = SFTConfig(
        max_seq_length              = MAX_SEQ_LENGTH,
        dataset_text_field          = "text",
        packing                     = False,

        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,

        num_train_epochs            = NUM_EPOCHS,
        learning_rate               = LEARNING_RATE,
        warmup_steps                = WARMUP_STEPS,
        weight_decay                = WEIGHT_DECAY,
        lr_scheduler_type           = LR_SCHEDULER,
        max_grad_norm               = MAX_GRAD_NORM,

        bf16                        = True,
        fp16                        = False,
        optim                       = OPTIMIZER,

        logging_steps               = LOGGING_STEPS,
        save_steps                  = SAVE_STEPS,
        save_total_limit            = 3,
        output_dir                  = OUTPUT_DIR,
        seed                        = SEED,
        report_to                   = "none",   # "wandb" if using W&B
        dataset_num_proc            = 4,
    ),
)


# ═══════════════════════════════════════════════════════════════
#  5. TRAIN ON COMPLETIONS ONLY
#     Masks all user / system tokens so loss is computed only on
#     assistant (model) responses.
# ═══════════════════════════════════════════════════════════════

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|turn>user\n",
    response_part = "<|turn>model\n",

)
# Verify masking: check that non-response tokens are masked (-100)
print("Masking verification:")
labels = trainer.train_dataset[0]["labels"]
masked_count  = sum(1 for l in labels if l == -100)
total_count   = len(labels)
trained_count = total_count - masked_count
print(f"  Masked tokens (instruction/user): {masked_count}")
print(f"  Trained tokens (assistant):       {trained_count}")
print(f"  Total tokens:                     {total_count}")
print(f"  Mask ratio:                       {masked_count / total_count:.1%}")

if trained_count == 0:
    raise ValueError(
        "⚠ ALL tokens are masked! train_on_responses_only found 0 assistant tokens.\n"
        "This means the instruction_part / response_part markers don't match your data.\n"
        "Check that your chat template produces <start_of_turn>model\\n markers."
    )

# Show what the model will actually train on (decode non-masked tokens)
print("\n--- Trained tokens preview (what the model learns) ---")
input_ids = trainer.train_dataset[0]["input_ids"]
trained_text = tokenizer.decode([t for t, l in zip(input_ids, labels) if l != -100])
print(trained_text[:500])
print("--- end preview ---\n")
# ═══════════════════════════════════════════════════════════════
#  6. TRAIN
# ═══════════════════════════════════════════════════════════════

print("=" * 60)
print("Starting training...")
print("=" * 60)

stats = trainer.train()

print("\n" + "=" * 60)
print("Training complete!")
print(f"  Total steps:   {stats.global_step}")
print(f"  Training loss: {stats.training_loss:.4f}")
print(f"  Runtime:       {stats.metrics['train_runtime']:.0f}s")
print("=" * 60)


# ═══════════════════════════════════════════════════════════════
#  7. SAVE — merged full weights (LoRA → base model merge)
# ═══════════════════════════════════════════════════════════════

print(f"\nMerging LoRA + saving full weights → {MERGED_SAVE_DIR}")
model.save_pretrained_merged(
    MERGED_SAVE_DIR,
    tokenizer,
    save_method="merged_16bit",
)
print("Merged weights saved.\n")


# ═══════════════════════════════════════════════════════════════
#  8. EXPORT TO GGUF
# ═══════════════════════════════════════════════════════════════

for quant in GGUF_QUANTS:
    save_path = os.path.join(GGUF_SAVE_DIR, quant)
    print(f"Exporting GGUF ({quant}) → {save_path}")
    model.save_pretrained_gguf(save_path, tokenizer, quantization_method=quant)
    print(f"  ✓ {quant} done.")

print("\n" + "=" * 60)
print("All done!")
print(f"  Checkpoints:     {OUTPUT_DIR}")
print(f"  Merged weights:  {MERGED_SAVE_DIR}")
print(f"  GGUFs:           {GGUF_SAVE_DIR}/[q4_k_m | q8_0 | f16]")
print("=" * 60)