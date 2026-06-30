#!/bin/bash
# Qwen-Style Dense SwiGLU Classifier Head
# Stage A: 只训练 TD Head (不开 LoRA)

set -e
cd /data2/wgy/qwen

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=========================================="
echo "Qwen-Style Dense Classifier Head Training"
echo "=========================================="
echo "Head: GQA (32H/4KVH, 128d) + Dense SwiGLU FFN (6144)"
echo "Layers: 1 decoder layer"
echo "Params: ~56M (dense head, no LoRA)"
echo "Training: Stage A (only classifier head)"
echo "Batch: 16, Grad-accum: 2"
echo "Output: ./train/output_stage_a_qwen_dense"
echo "=========================================="

# 清理 GPU 显存
ps aux | grep "train/train.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
sleep 3

conda run -n qwen --no-capture-output python train/train.py \
    --model_path ./Qwen3-Omni-30B-A3B-Instruct \
    --output_dir ./train/output_stage_a_qwen_dense \
    --train_list_file ./dataset/Easy-Turn/Trainset_list/merged_balanced.list \
    --trainset_dir ./dataset/Easy-Turn/Trainset \
    --eval_list_file ./dataset/Easy-Turn/Testset/testset_all.list \
    --evalset_dir ./dataset/Easy-Turn/Testset \
    --sampler none \
    --stage A \
    --td_head_type qwen \
    --ffn_type dense \
    --num_td_layers 1 \
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
    2>&1 | tee ./train/output_stage_a_qwen_dense/train.log

echo ""
echo "✅ Dense Classifier Head training complete"
echo "📁 Model: ./train/output_stage_a_qwen_dense/"
echo "📊 Log: ./train/output_stage_a_qwen_dense/train.log"