#!/bin/bash
# 阶段 A: 只训练 TD Head (MLP 版本)
# 8×3090 (24GB) 配置

set -e
cd /data2/wgy/qwen

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "阶段 A: 只训练 TD Head (MLP 版本)"
echo "=========================================="
echo "模型: FP16 + device_map=auto (分布到 8 卡)"
echo "训练: 只训练 TD Head (MLP)，不开 LoRA"
echo "训练集: merged_balanced.list (304k 样本)"
echo "评估集: testset_all.list (800 样本)"
echo "输出目录: ./train/output_stage_a_mlp"
echo "=========================================="

# 清理 GPU 显存
ps aux | grep "train/train.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
sleep 3

python train/train.py \
    --model_path ./Qwen3-Omni-30B-A3B-Instruct \
    --output_dir ./train/output_stage_a_mlp \
    --train_list_file ./dataset/Easy-Turn/Trainset_list/merged_balanced.list \
    --trainset_dir ./dataset/Easy-Turn/Trainset \
    --eval_list_file ./dataset/Easy-Turn/Testset/testset_all.list \
    --evalset_dir ./dataset/Easy-Turn/Testset \
    --sampler weighted \
    --stage A \
    --td_head_type mlp \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --max_audio_seconds 60 \
    --lr_td_head 5e-4 \
    --logging_steps 50 \
    --eval_steps 200 \
    --save_steps 500 \
    --early_stopping_patience 10 \
    --num_workers 16 \
    2>&1 | tee ./train/output_stage_a_mlp/train.log

echo ""
echo "✅ 阶段 A (MLP) 训练完成"
echo "📁 模型: ./train/output_stage_a_mlp/"
echo "📊 日志: ./train/output_stage_a_mlp/train.log"
