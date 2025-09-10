
for sl in 1024 2048 4096
do
    k=$((sl / 1024))
    mkdir -p log/llama3/yb/${k}k
    for gpus in 4 8 16 32
    do
        c=$((gpus / 4))
        export TORCH_SDAA_ALLOC_CONF=max_split_size_mb:300
        deepspeed --master_port 29001 --num_gpus=$gpus pipeline/llama.py \
            --deepspeed config/zero_config/ds_config_zero3_llama.json \
            --model_name_or_path /data02/application/zhuzh/workspace/download/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6/ \
            --dataset_name nisaar/Articles_Constitution_3300_Instruction_Set \
            --per_device_train_batch_size 1 \
            --logging_steps 1 \
            --do_train \
            --gradient_checkpointing=True \
            --fp16 True \
            --max_steps 100 \
            --block_size ${sl} \
            --overwrite_output_dir \
            --output_dir output 2>&1 | tee log/llama3_sdaa_${k}k_${c}c_1.log
    done
done


# for sl in 1024 2048 4096
# do
#     k=$((sl / 1024))
#     mkdir -p log/llama3/tb/${k}k
#     for gpus in 4 8 16 32
#     do
#         c=$((gpus / 4))
#         export TORCH_SDAA_ALLOC_CONF=max_split_size_mb:300
#         TECO_LAUNCH_BLOCKING=1 SDAA_LAUNCH_BLOCKING=1 \
#         deepspeed --master_port 29000 --num_gpus=$gpus llama2_new.py \
#             --deepspeed config/qwen2/ds_config_zero3.json \
#             --model_name_or_path /data02/application/zhuzh/workspace/download/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6/ \
#             --dataset_name nisaar/Articles_Constitution_3300_Instruction_Set \
#             --per_device_train_batch_size 1 \
#             --logging_steps 1 \
#             --do_train \
#             --gradient_checkpointing=True \
#             --fp16 True \
#             --max_steps 100 \
#             --block_size ${sl} \
#             --overwrite_output_dir \
#             --output_dir output 2>&1 | tee log/llama3/tb/${k}k/sdaa_${k}k_${c}c.log
#     done
# done
