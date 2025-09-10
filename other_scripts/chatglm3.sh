# TORCH_SDAA_LOG_LEVEL=debug
# TECO_ENABLE_DUMP_INFO=1 \
# TECO_LAUNCH_BLOCKING=1 \
# SDAA_LAUNCH_BLOCKING=1 \

export TORCH_SDAA_ALLOC_CONF=max_split_size_mb:300
# deepspeed --master_port 29000 --include localhost:16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 chatglm3.py \
# SDAA_VISIBLE_DEVICES=16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \
torchrun --standalone --nnodes=1 --nproc_per_node=32 chatglm3.py \
    # datasets/chatglm3_6b \
    # /data/application/zhuzh/workspace/download/models--THUDM--chatglm3-6b-1/snapshots/06c7c873c843814171c51330b69c2e2a68e05178 \
    # config/chatglm3_6b.yaml
