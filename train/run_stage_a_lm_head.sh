#!/bin/bash
# LM Head TD - 基于预训练 LM Head 的分类头
# Stage A: 只训练 TD Head (不开 LoRA)
# 使用预训练 LM Head 权重初始化

set -e
cd /data2/wgy/qwen

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "LM Head TD Training (Stage A)"
echo "=========================================="
echo "架构: 双 LM Head"
echo "  - ASR Head: 保持预训练权重 (不训练)"
echo "  - TD Head: 基于预训练权重, 输出 4 分类"
echo "TD Head: Attention Pooling + Linear(2048, 4)"
echo "初始化: 从预训练 LM Head 采样 4 行权重"
echo "Batch: 32, Grad-accum: 4"
echo "Output: ./train/output_stage_a_lm_head"
echo "=========================================="

# 清理 GPU 显存
ps aux | grep "train/train.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
sleep 3

conda run -n qwen --no-capture-output python train/train.py \
    --model_path ./Qwen3-Omni-30B-A3B-Instruct \
    --output_dir ./train/output_stage_a_lm_head \
    --train_list_file ./dataset/Easy-Turn/Trainset_list/merged_balanced.list \
    --trainset_dir ./dataset/Easy-Turn/Trainset \
    --eval_list_file ./dataset/Easy-Turn/Testset/testset_all.list \
    --evalset_dir ./dataset/Easy-Turn/Testset \
    --sampler none \
    --stage A \
    --td_head_type lm_head \
    --num_train_epochs 2 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --max_audio_seconds 60 \
    --lr_td_head 5e-4 \
    --logging_steps 50 \
    --eval_steps 200 \
    --save_steps 500 \
    --early_stopping_patience 10 \
    --num_workers 32 \
    --class_weights 1.0 1.5 3.0 3.0 \
    2>&1 | tee ./train/output_stage_a_lm_head/train.log

echo ""
echo "✅ LM Head TD training complete"
echo "📁 Model: ./train/output_stage_a_lm_head/"
echo "📊 Log: ./train/output_stage_a_lm_head/train.log"
