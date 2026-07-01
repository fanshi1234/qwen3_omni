#!/bin/bash
# Token TD Head - 与原始 LM Head 完全一致的分类头
# Stage A: 只训练 TD Head (不开 LoRA)
# 使用列向量平均值 + 缩放初始化

set -e
cd /data2/wgy/qwen

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "Token TD Head Training (Stage A)"
echo "=========================================="
echo "架构: 与原始 LM Head 完全一致"
echo "  - 输入: 所有 hidden states [B, T, 2048]"
echo "  - 分类层: Linear(2048, 4, bias=False)"
echo "  - 取最后有效位置的 logits"
echo "初始化: 列向量平均值 + 缩放"
echo "  - Complete: complete, finish, end, done, over, conclude, 完成, 结束, 搞定, 收尾, 完结"
echo "  - InComplete: incomplete, continue, keep, go, proceed, next, 继续, 接着, 未完, 待续, 进行"
echo "  - Backchannel: backchannel, mm, uh, yeah, okay, right, 嗯, 哦, 好的, 对, 啊"
echo "  - Wait: wait, interrupt, hold, stop, pause, halt, 稍等, 暂停, 停下, 等等, 打断"
echo "学习率: 5e-4"
echo "梯度裁剪: 1.0"
echo "weight_decay: 0.01"
echo "初始化策略: column (列向量平均值)"
echo "缩放系数: 0.5"
echo "Batch: 32, Grad-accum: 4"
echo "Output: ./train/output_stage_a_token"
echo "=========================================="

# 清理 GPU 显存
ps aux | grep "train/train.py" | grep "token" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
sleep 3

conda run -n qwen --no-capture-output python train/train.py \
    --model_path ./Qwen3-Omni-30B-A3B-Instruct \
    --output_dir ./train/output_stage_a_token \
    --train_list_file ./dataset/Easy-Turn/Trainset_list/merged_balanced.list \
    --trainset_dir ./dataset/Easy-Turn/Trainset \
    --eval_list_file ./dataset/Easy-Turn/Testset/testset_all.list \
    --evalset_dir ./dataset/Easy-Turn/Testset \
    --sampler none \
    --stage A \
    --td_head_type token \
    --init_strategy column \
    --init_scale 0.5 \
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
    2>&1 | tee ./train/output_stage_a_token/train.log

echo ""
echo "✅ Token TD Head training complete"
echo "📁 Model: ./train/output_stage_a_token/"
echo "📊 Log: ./train/output_stage_a_token/train.log"
