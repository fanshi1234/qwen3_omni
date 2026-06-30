#!/bin/bash
# 阶段 B: TD Head (MLP) + QLoRA
# 8×3090 (24GB) 配置

set -e
cd /data2/wgy/qwen

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "阶段 B: TD Head (MLP) + QLoRA"
echo "=========================================="
echo "加载第一阶段最优权重: ./train/output_stage_a_mlp/best_checkpoint/td_head.pt"
echo "Batch size: 8, 梯度累积: 16, 最大音频: 20秒"
echo "输出目录: ./train/output_stage_b_mlp"
echo "=========================================="

# 清理 GPU 显存
ps aux | grep "train/train.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
sleep 3

python train/train.py \
    --model_path ./Qwen3-Omni-30B-A3B-Instruct \
    --output_dir ./train/output_stage_b_mlp \
    --train_list_file ./dataset/Easy-Turn/Trainset_list/merged_balanced.list \
    --trainset_dir ./dataset/Easy-Turn/Trainset \
    --eval_list_file ./dataset/Easy-Turn/Testset/testset_all.list \
    --evalset_dir ./dataset/Easy-Turn/Testset \
    --sampler weighted \
    --stage B \
    --td_head_type mlp \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --max_audio_seconds 20 \
    --lr_td_head 5e-4 \
    --lr_lora 5e-5 \
    --logging_steps 50 \
    --eval_steps 200 \
    --save_steps 500 \
    --early_stopping_patience 10 \
    --num_workers 32 \
    --load_td_head ./train/output_stage_a_mlp/best_checkpoint/td_head.pt \
    2>&1 | tee ./train/output_stage_b_mlp/train.log

echo ""
echo "✅ 阶段 B (MLP) 训练完成"
echo "📁 模型: ./train/output_stage_b_mlp/"
echo "📊 日志: ./train/output_stage_b_mlp/train.log"
