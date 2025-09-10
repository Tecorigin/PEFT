# TECO_ENABLE_DUMP_INFO=1 \
# TECO_LAUNCH_BLOCKING=1 \
# SDAA_LAUNCH_BLOCKING=1 \
# SDAA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \

for sl in 1024 # 2048 4096
do
    k=$((sl / 1024))
    mkdir -p log/chatglm3/yb/${k}k
    for gpus in 4
    do
        c=$((gpus / 4))
        export OMP_NUM_THREADS=4
        export TORCH_SDAA_LINEAR_HIGHPREC=1
        export TORCH_SDAA_BADDBMM_HIGHPREC=1
        export TORCH_SDAA_BMM_HIGHPREC=1
        export TORCH_SDAA_BMM_HIGHPERF=1
        export TORCH_SDAA_BLAS_TRANSPOSE=0
        export TORCH_SDAA_FUSED_ATTN_MEM_LIMITED=1
        export TORCH_SDAA_ALIGN_NV_DEVICE=a100
        export TORCH_SDAA_DISABLE_FLASH_SDPA=1
        deepspeed --master_port 29000 --num_gpus=$gpus llama2_new.py \
            --deepspeed config/qwen2/ds_config_zero3.json \
            --model_name_or_path /data02/application/zhuzh/workspace/download/models--THUDM--chatglm3-6b/snapshots/06c7c873c843814171c51330b69c2e2a68e05178 \
            --dataset_name nisaar/Articles_Constitution_3300_Instruction_Set \
            --trust_remote_code=True \
            --per_device_train_batch_size 1 \
            --do_train \
            --gradient_checkpointing=True \
            --fp16 True \
            --max_steps 100 \
            --block_size ${sl} \
            --overwrite_output_dir \
            --output_dir output 2>&1 | tee log/chatglm3/yb/${k}k/sdaa_${k}k_${c}c.log
    done
done


# TECO_ENABLE_DUMP_INFO=1 \
# TECO_LAUNCH_BLOCKING=1 \
# SDAA_LAUNCH_BLOCKING=1 \
# SDAA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \

# for sl in 1024 2048 4096
# do
#     k=$((sl / 1024))
#     mkdir -p log/chatglm3/tb/${k}k
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
#             --do_train \
#             --gradient_checkpointing=True \
#             --fp16 True \
#             --max_steps 100 \
#             --block_size ${sl} \
#             --overwrite_output_dir \
#             --output_dir output 2>&1 | tee log/chatglm3/tb/${k}k/sdaa_${k}k_${c}c.log
#     done
# done
