#!/bin/bash
# UAF 双头训练 - 快速启动脚本
# 根据 stage 参数选择不同配置

set -e
cd /data1/wgy/qwen

STAGE=${1:-"A"}  # 默认 Stage A

echo "=========================================="
echo "UAF Turn Head Training"
echo "Stage: $STAGE"
echo "=========================================="

case $STAGE in
    A|a)
        echo "📋 Stage A: 只训练 TD Head"
        bash train/run_stage_a.sh
        ;;
    B|b)
        echo "📋 Stage B: QLoRA + TD Head + ASR"
        bash train/run_stage_b.sh
        ;;
    *)
        echo "❌ 未知 Stage: $STAGE"
        echo "用法: bash train/run.sh [A|B]"
        exit 1
        ;;
esac
