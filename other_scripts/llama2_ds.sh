output_model=llama2_output
if [ ! -d ${output_model} ];then
    mkdir ${output_model}
fi
deepspeed --master_port 29000 --num_gpus=1  llama2_ds.py \
    --model_name_or_path /data/application/zhuzh/workspace/download/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
    --tokenizer_name /data/application/zhuzh/workspace/download/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9 \
    --train_files /data/application/zhuzh/workspace/q3/peft/peft-main/datasets/llama2/alpaca_gpt4_data_zh.json \
    --validation_files /data/application/zhuzh/workspace/q3/peft/peft-main/datasets/llama2/trans_chinese_alpaca_data.json \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_fast_tokenizer true \
    --output_dir ${output_model} \
    --evaluation_strategy  steps \
    --max_eval_samples 800 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 5 \
    --warmup_steps 400 \
    --lora_r 8 \
    --lora_alpha 16 \
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --preprocessing_num_workers 10 \
    --save_steps 200 \
    --eval_steps 200 \
    --save_total_limit 2000 \
    --torch_dtype float32 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 1024 \
    --report_to tensorboard \
    --overwrite_output_dir \
    --ignore_data_skip true \
    --gradient_checkpointing \
    --ddp_timeout 18000000 \
    --max_steps 1 \
    --deepspeed /data/application/zhuzh/workspace/q3/peft/peft-main/config/qwen2/ds_config_zero2.json
