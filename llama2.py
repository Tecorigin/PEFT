# Import necessary packages for the fine-tuning process
import os                          # Operating system functionalities
import torch                       # PyTorch library for deep learning
from datasets import load_dataset  # Loading datasets for training
from transformers import (
    AutoModelForCausalLM,          # AutoModel for language modeling tasks
    AutoTokenizer,                # AutoTokenizer for tokenization
    BitsAndBytesConfig,           # Configuration for BitsAndBytes
    HfArgumentParser,             # Argument parser for Hugging Face models
    TrainingArguments,            # Training arguments for model training
    pipeline,                     # Creating pipelines for model inference
    logging,                      # Logging information during training
)
from peft import LoraConfig, PeftModel  # Packages for parameter-efficient fine-tuning (PEFT)
from trl import SFTTrainer         # SFTTrainer for supervised fine-tuning
# The model that you want to train from the Hugging Face hub
model_name = "/data/application/zhuzh/workspace/download/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
# model_name="/data/application/zhuzh/workspace/download/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6"
# The instruction dataset to use
dataset_name = "mlabonne/guanaco-llama2-1k"

# Fine-tuned model name
new_model = "llama-2-7b-miniguanaco"
# LoRA attention dimension
lora_r = 8

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1
# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"

# Number of training epochs
num_train_epochs = 1

# Enable fp16/bf16 training (set bf16 to True with an A100)
fp16 = False
bf16 = False

# Batch size per GPU for training
per_device_train_batch_size = 1

# Batch size per GPU for evaluation
per_device_eval_batch_size = 1

# Number of update steps to accumulate the gradients for
gradient_accumulation_steps = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Maximum gradient normal (gradient clipping)
max_grad_norm = 0.3

# Initial learning rate (AdamW optimizer)
learning_rate = 2e-4

# Weight decay to apply to all layers except bias/LayerNorm weights
weight_decay = 0.001

# Optimizer to use
optim = "adamw_torch"

# Learning rate schedule (constant a bit better than cosine)
lr_scheduler_type = "constant"

# Number of training steps (overrides num_train_epochs)
max_steps = 1

# Ratio of steps for a linear warmup (from 0 to learning rate)
warmup_ratio = 0.03

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
group_by_length = True

# Save checkpoint every X updates steps
save_steps = 25

# Log every X updates steps
logging_steps = 25
max_seq_length = 4096

# Pack multiple short examples in the same input sequence to increase efficiency
packing = False

# Load the entire model on the GPU 0
device_map = {"sdaa": 0}
dataset = load_dataset(dataset_name, split="train")
bnb_4bit_compute_dtype = "float16"
use_4bit = True
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
# if compute_dtype == torch.float16 and use_4bit:
#     import torch_sdaa
#     major, _ = torch_sdaa.get_device_capability()
#     if major >= 8:
#         print("=" * 80)
#         print("Your GPU supports bfloat16: accelerate training with bf16=True")
#         print("=" * 80)
# Step 4 :Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    ignore_mismatched_sizes=True
)
model.config.use_cache = False
model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
    deepspeed="/data/application/zhuzh/workspace/q3/peft/peft-main/config/qwen2/ds_config_zero3.json"
)
print(training_arguments)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)
trainer.train()
# trainer.model.save_pretrained(new_model)
