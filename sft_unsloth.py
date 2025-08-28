import os

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

import wandb
from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="model_name",
    max_seq_length=4096,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)


train_dataset = load_dataset("parquet", data_files="path/to/train.parquet")
test_dataset = load_dataset("parquet", data_files="path/to/test.parquet")


trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,  # Can set up evaluation!
    args=SFTConfig(
        max_seq_length=4096,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Use GA to mimic batch size!
        warmup_steps=5,
        num_train_epochs=1,  # Set this for 1 full training run.
        # max_steps = 30,
        learning_rate=5e-6,  # Reduce to 2e-5 for long training runs
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="wandb",  # Use this for WandB etc,
        save_steps=100,
        save_strategy="steps",
        eval_steps=10,
        eval_strategy="steps",
        dataset_text_field="text",
    ),
)

wandb.init(project="unsloth-sft", entity="L16H7")
trainer_stats = trainer.train()
