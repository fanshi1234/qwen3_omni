"""
分析 Wait 类别的误判情况
"""

import os
import sys
import json
import torch
import logging
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SYSTEM_PROMPT, LABEL2ID, ID2STATE
from dataset import TurnDataset, collate_fn
from td_head import TDHead
from train import UAFModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_errors(model, eval_list_file, evalset_dir, max_audio_seconds, batch_size=4):
    """分析误判样本"""
    eval_dataset = TurnDataset(
        list_file=eval_list_file,
        trainset_dir=evalset_dir,
        max_duration=max_audio_seconds,
    )

    model.eval()

    # 统计
    total = 0
    correct = 0
    errors = defaultdict(list)  # {true_label: [(pred, key, txt, wav), ...]}
    confusion = defaultdict(lambda: defaultdict(int))  # {true: {pred: count}}

    with torch.no_grad():
        for i in tqdm(range(len(eval_dataset)), desc="Analyzing"):
            item = eval_dataset[i]

            # 获取预测
            outputs = model(
                audio_list=[item['audio_path']],
                text_targets=[item['text_target']],
            )
            pred = torch.argmax(outputs["turn_logits"], dim=-1).cpu().numpy()[0]
            true_label = item['turn_label']

            # 统计
            total += 1
            if pred == true_label:
                correct += 1
            else:
                errors[true_label].append({
                    'pred': pred,
                    'key': item.get('key', 'unknown'),
                    'txt': item.get('text_target', ''),
                    'wav': item.get('audio_path', ''),
                    'duration': item.get('duration', 0),
                })

            confusion[true_label][pred] += 1

    return total, correct, errors, confusion


def print_confusion_matrix(confusion):
    """打印混淆矩阵"""
    labels = ['Complete', 'InComplete', 'Backchannel', 'Wait']
    print("\n" + "="*80)
    print("混淆矩阵 (行=真实, 列=预测)")
    print("="*80)

    # 打印表头
    header = "真实\\预测"
    print(f"{header:<15}", end="")
    for label in labels:
        print(f"{label:<12}", end="")
    print()

    # 打印每一行
    for i, true_label in enumerate(labels):
        print(f"{true_label:<15}", end="")
        for j, pred_label in enumerate(labels):
            count = confusion[i].get(j, 0)
            print(f"{count:<12}", end="")
        print()


def print_error_analysis(errors, top_n=20):
    """打印误判分析"""
    labels = ['Complete', 'InComplete', 'Backchannel', 'Wait']

    print("\n" + "="*80)
    print("误判分析")
    print("="*80)

    for true_label_id, error_list in errors.items():
        true_label = labels[true_label_id]
        print(f"\n{'='*60}")
        print(f"真实类别: {true_label} (误判 {len(error_list)} 个)")
        print(f"{'='*60}")

        # 统计误判到哪些类别
        pred_counter = Counter([e['pred'] for e in error_list])
        print(f"误判分布:")
        for pred_id, count in pred_counter.most_common():
            pred_label = labels[pred_id]
            print(f"  -> {pred_label}: {count} ({count/len(error_list)*100:.1f}%)")

        # 打印部分误判样本
        print(f"\n部分误判样本 (前 {min(top_n, len(error_list))} 个):")
        for j, error in enumerate(error_list[:top_n]):
            pred_label = labels[error['pred']]
            print(f"  [{j+1}] {error['key']}")
            print(f"      预测: {pred_label}")
            print(f"      文本: {error['txt'][:100]}...")
            print(f"      时长: {error['duration']:.2f}s")
            print(f"      音频: {error['wav']}")
            print()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Wait class errors")
    parser.add_argument("--model_path", type=str, default="/data2/wgy/qwen/Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--checkpoint_path", type=str,
                        default="/data2/wgy/qwen/train/output_stage_b/best_checkpoint")
    parser.add_argument("--eval_list_file", type=str,
                        default="/data2/wgy/qwen/dataset/Easy-Turn/Testset/testset_all.list")
    parser.add_argument("--evalset_dir", type=str,
                        default="/data2/wgy/qwen/dataset/Easy-Turn/Testset")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_audio_seconds", type=float, default=60.0)

    args = parser.parse_args()

    logger.info("="*60)
    logger.info("分析 Wait 类别误判情况")
    logger.info("="*60)
    logger.info(f"基座模型: {args.model_path}")
    logger.info(f"Checkpoint: {args.checkpoint_path}")
    logger.info(f"评估集: {args.eval_list_file}")
    logger.info("="*60)

    # 初始化模型（不添加 LoRA）
    model = UAFModel(
        model_path=args.model_path,
        training_stage="A",
        use_lora=False,
    )

    # 加载 LoRA 权重
    lora_path = os.path.join(args.checkpoint_path, "adapter_model.safetensors")
    if os.path.exists(lora_path):
        logger.info(f"Loading LoRA weights from {lora_path}")
        # 使用 PeftModel.from_pretrained 正确加载 LoRA
        from peft import PeftModel
        model.base_model = PeftModel.from_pretrained(
            model.base_model,
            args.checkpoint_path,
            is_trainable=False,
        )
        logger.info("LoRA weights loaded successfully")

    # 加载 TD Head 权重
    td_head_path = os.path.join(args.checkpoint_path, "td_head.pt")
    if os.path.exists(td_head_path):
        logger.info(f"Loading TD Head weights from {td_head_path}")
        model.td_head.load_state_dict(torch.load(td_head_path, map_location="cpu"))
        logger.info("TD Head weights loaded successfully")

    # 运行分析
    logger.info("\n开始分析...")
    total, correct, errors, confusion = analyze_errors(
        model,
        args.eval_list_file,
        args.evalset_dir,
        args.max_audio_seconds,
        batch_size=args.batch_size,
    )

    # 打印结果
    print("\n" + "="*80)
    print("分析结果汇总")
    print("="*80)
    print(f"总样本数: {total}")
    print(f"正确数: {correct}")
    print(f"错误数: {total - correct}")
    print(f"准确率: {correct/total*100:.2f}%")

    # 打印混淆矩阵
    print_confusion_matrix(confusion)

    # 打印误判分析
    print_error_analysis(errors)


if __name__ == "__main__":
    main()
