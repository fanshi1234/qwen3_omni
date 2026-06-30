#!/bin/bash
# 阶段 B: TD Head + QLoRA (使用类别权重，不使用采样)
# 8×3090 (24GB) 配置

set -e
cd /data2/wgy/qwen

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "阶段 B: TD Head + QLoRA (类别权重)"
echo "=========================================="
echo "加载第一阶段最优权重: ./train/output_stage_a_cls/best_checkpoint/td_head.pt"
echo "采样: 不使用采样 (sampler=none)"
echo "类别权重: [1.0, 1.5, 3.0, 3.0]"
echo "Batch size: 8, 梯度累积: 16, 最大音频: 20秒"
echo "输出目录: ./train/output_stage_b_cls"
echo "=========================================="

# 清理 GPU 显存
ps aux | grep "train/train.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
sleep 3

python train/train.py \
    --model_path ./Qwen3-Omni-30B-A3B-Instruct \
    --output_dir ./train/output_stage_b_cls \
    --train_list_file ./dataset/Easy-Turn/Trainset_list/merged_balanced.list \
    --trainset_dir ./dataset/Easy-Turn/Trainset \
    --eval_list_file ./dataset/Easy-Turn/Testset/testset_all.list \
    --evalset_dir ./dataset/Easy-Turn/Testset \
    --sampler none \
    --stage B \
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
    --load_td_head ./train/output_stage_a_cls/best_checkpoint/td_head.pt \
    --class_weights 1.0 1.5 3.0 3.0 \
    2>&1 | tee ./train/output_stage_b_cls/train.log

echo ""
echo "✅ 阶段 B (类别权重) 训练完成"
echo "📁 模型: ./train/output_stage_b_cls/"
echo "📊 日志: ./train/output_stage_b_cls/train.log"
