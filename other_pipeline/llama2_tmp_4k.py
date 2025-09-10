from datasets import load_dataset

dataset_name = 'nisaar/Articles_Constitution_3300_Instruction_Set'
dataset = load_dataset(dataset_name, split="train")
from datasets import Dataset

# Assuming `dataset` is your Dataset object
dataset = dataset.map(lambda example: {'text': example['prompt'] + example['output']})
import torch,torch_sdaa
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from transformers import DataCollatorForSeq2Seq
model_name = "/data02/application/zhuzh/workspace/download/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9/"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
)
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
from peft import LoraConfig, get_peft_model

lora_alpha = 8
lora_dropout = 0.1
lora_r = 8

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    # target_modules=['q_proj','k_proj','v_proj','o_proj','down_proj','gate_proj','up_proj'],
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)
from transformers import TrainingArguments
output_dir = "./results"
per_device_train_batch_size = 1
gradient_accumulation_steps = 2
save_steps = 1
num_train_epochs = 4
logging_steps = 1
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 100
warmup_ratio = 0.03
lr_scheduler_type = "linear"
# deepspeed = "config/qwen2/ds_config_zero3.json"
# print(deepspeed)
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    deepspeed="config/qwen2/ds_config_zero3.json",
    gradient_checkpointing=True
)
from trl import SFTTrainer
max_seq_length = 4096
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    packing=True,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)
# for name, module in trainer.model.named_modules():
#     if "norm" in name:
#         module = module.to(torch.float32)

trainer.train()
