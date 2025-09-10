# TECO_ENABLE_DUMP_INFO=1 \
# TECO_LAUNCH_BLOCKING=1 \
# SDAA_LAUNCH_BLOCKING=1 \
# TORCH_SDAA_LOG_LEVEL=debug \
# SDAA_VISIBLE_DEVICES=0,1,2,3 \

# deepspeed --num_gpus 4 --master_port=9901 qwen2.py
#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training).
# Please set the options below according to the comments.
# For multi-gpu workers training, these options should be manually set for each worker.
# After setting the options, please run the script on each worker.

# Number of GPUs per GPU worker
GPUS_PER_NODE=$(python -c 'import torch_sdaa,torch; print(torch.sdaa.device_count())')
GPUS_PER_NODE=4
# Number of GPU workers, for single-worker training, please set to 1
NNODES=${NNODES:-1}

# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
NODE_RANK=${NODE_RANK:-0}

# The ip address of the rank-0 worker, for single-worker training, please set to localhost
MASTER_ADDR=${MASTER_ADDR:-localhost}

# The port for communication
MASTER_PORT=${MASTER_PORT:-6001}

MODEL="Qwen/Qwen2-7B" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See https://qwen.readthedocs.io/en/latest/training/SFT/example.html#data-preparation for more information.
DATA="example_data.jsonl"
DS_CONFIG_PATH="ds_config_zero3.json"
USE_LORA=False
Q_LORA=False
MAX_LENGTH=2048
function usage() {
    echo '
Usage: bash finetune.sh [-m MODEL_PATH] [-d DATA_PATH] [--deepspeed DS_CONFIG_PATH] [--use_lora USE_LORA] [--q_lora Q_LORA]
'
}

while [[ "$1" != "" ]]; do
    case $1 in
        -m | --model )
            shift
            MODEL=$1
            ;;
        -l | --max_length )
            shift
            MAX_LENGTH=$1
            ;;
        -d | --data )
            shift
            DATA=$1
            ;;
        --deepspeed )
            shift
            DS_CONFIG_PATH=$1
            ;;
        --use_lora  )
            shift
            USE_LORA=$1
            ;;
        --q_lora    )
            shift
            Q_LORA=$1
            ;;
        -h | --help )
            usage
            exit 0
            ;;
        * )
            echo "Unknown argument ${1}"
            exit 1
            ;;
    esac
    shift
done

DISTRIBUTED_ARGS="
    --nproc_per_node 8 \
    --nnodes 1 \
    --master_port 9901
"

echo $MAX_LENGTH
echo $DS_CONFIG_PATH

# TORCH_SDAA_ALLOC_CONF=max_split_size_mb:600 \
# TORCH_SDAA_LOG_LEVEL=debug \
# TECO_ENABLE_DUMP_INFO=1 \
# TECO_LAUNCH_BLOCKING=1 \
# SDAA_LAUNCH_BLOCKING=1 \
# export TORCH_SDAA_RUNTIME_AUTOFALLBACK=1
# export TORCH_SDAA_FALLBACK_OPS=_log_softmax,_log_softmax_backward_data,_softmax,add,add_,cos,embedding,eq,gt,index,masked_fill_,mul,mul_,nll_loss_backward,nll_loss_forward,reciprocal,rsqrt,sin,slice_backward,dropout
for sl in 1024 #2048 4096
do
    k=$((sl / 1024))
    # mkdir -p log/qwen2-7b/${k}k
    for gpus in 32
    do
        c=$((gpus / 4))
        export TORCH_SDAA_ALLOC_CONF=max_split_size_mb:300
        deepspeed --num_gpus=$gpus --master_port 9901 qwen2_finetune.py \
                --model_name_or_path /data02/application/zhuzh/workspace/download/models--Qwen--Qwen2-7B-Instruct/snapshots/41c66b0be1c3081f13defc6bdf946c2ef240d6a6 \
                --data_path config/qwen2/example_data.jsonl \
                --bf16 False \
                --fp16 True \
                --output_dir output_qwen \
                --num_train_epochs 1 \
                --per_device_train_batch_size 1 \
                --max_steps=100 \
                --per_device_eval_batch_size 1 \
                --gradient_accumulation_steps 8 \
                --gradient_checkpointing True \
                --evaluation_strategy "no" \
                --save_strategy "steps" \
                --save_steps 10 \
                --save_total_limit 10 \
                --learning_rate 3e-4 \
                --weight_decay 0.01 \
                --adam_beta2 0.95 \
                --warmup_ratio 0.01 \
                --lr_scheduler_type "cosine" \
                --logging_steps 1 \
                --report_to "none" \
                --model_max_length ${sl} \
                --lazy_preprocess True \
                --use_lora True \
                --q_lora False \
                --deepspeed config/qwen2/ds_config_zero2.json 2>&1 | tee log/qwen2-7b/yb/${k}k/sdaa_${k}k_${c}c.log
    done
done

for sl in 1024 2048 4096
do
    k=$((sl / 1024))
    # mkdir -p log/qwen2-7b/tb/${k}k
    for gpus in 32
    do
        c=$((gpus / 4))
        export TORCH_SDAA_ALLOC_CONF=max_split_size_mb:300
        TECO_LAUNCH_BLOCKING=1 SDAA_LAUNCH_BLOCKING=1 \
        deepspeed --num_gpus=$gpus --master_port 9901 qwen2_finetune.py \
                --model_name_or_path /data02/application/zhuzh/workspace/download/models--Qwen--Qwen2-7B-Instruct/snapshots/41c66b0be1c3081f13defc6bdf946c2ef240d6a6 \
                --data_path config/qwen2/example_data.jsonl \
                --bf16 False \
                --fp16 True \
                --output_dir output_qwen \
                --num_train_epochs 1 \
                --per_device_train_batch_size 1 \
                --max_steps=100 \
                --per_device_eval_batch_size 1 \
                --gradient_accumulation_steps 8 \
                --gradient_checkpointing True \
                --evaluation_strategy "no" \
                --save_strategy "steps" \
                --save_steps 10 \
                --save_total_limit 10 \
                --learning_rate 3e-4 \
                --weight_decay 0.01 \
                --adam_beta2 0.95 \
                --warmup_ratio 0.01 \
                --lr_scheduler_type "cosine" \
                --logging_steps 1 \
                --report_to "none" \
                --model_max_length ${sl} \
                --lazy_preprocess True \
                --use_lora True \
                --q_lora False \
                --deepspeed config/qwen2/ds_config_zero3.json 2>&1 | tee log/qwen2-7b/tb/${k}k/sdaa_${k}k_${c}c.log
    done
done
