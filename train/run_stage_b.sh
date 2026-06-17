#!/bin/bash
# 阶段 B: TD Head + QLoRA
# 8×3090 (24GB) 配置
# 加载第一阶段最优 TD Head 权重

set -e
cd /data2/wgy/qwen

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "阶段 B: TD Head + QLoRA"
echo "=========================================="
echo "加载第一阶段最优权重: ./train/output_stage_a/best_checkpoint/td_head.pt"
echo "=========================================="

# 清理 GPU 显存
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null
sleep 3

python train/train.py \
    --model_path ./Qwen3-Omni-30B-A3B-Instruct \
    --output_dir ./train/output_stage_b \
    --train_list_file ./dataset/Easy-Turn/Trainset_list/merged_balanced.list \
    --trainset_dir ./dataset/Easy-Turn/Trainset \
    --eval_list_file ./dataset/Easy-Turn/Testset/testset_all.list \
    --evalset_dir ./dataset/Easy-Turn/Testset \
    --sampler weighted \
    --stage B \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --max_audio_seconds 10 \
    --lr_td_head 5e-4 \
    --lr_lora 5e-5 \
    --logging_steps 50 \
    --eval_steps 200 \
    --save_steps 500 \
    --early_stopping_patience 10 \
    --load_td_head ./train/output_stage_a/best_checkpoint/td_head.pt \
    2>&1 | tee ./train/output_stage_b/train.log

echo ""
echo "✅ 阶段 B 训练完成"
echo "📁 模型: ./train/output_stage_b/"
echo "📊 日志: ./train/output_stage_b/train.log"
