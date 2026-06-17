"""
加载一阶段最优模型进行评估
"""

import os
import sys
import argparse
import torch
import logging
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SYSTEM_PROMPT, LABEL2ID, ID2STATE
from dataset import TurnDataset, collate_fn
from td_head import TDHead
from train import UAFModel, compute_metrics, log_metrics, evaluate_on_testset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate best checkpoint from Stage A")
    parser.add_argument("--model_path", type=str, default="/data2/wgy/qwen/Qwen3-Omni-30B-A3B-Instruct",
                        help="Path to base model")
    parser.add_argument("--checkpoint_path", type=str,
                        default="/data2/wgy/qwen/train/output_stage_a/best_checkpoint",
                        help="Path to best checkpoint")
    parser.add_argument("--eval_list_file", type=str,
                        default="/data2/wgy/qwen/dataset/Easy-Turn/Testset/testset_all.list",
                        help="Path to evaluation list file")
    parser.add_argument("--evalset_dir", type=str,
                        default="/data2/wgy/qwen/dataset/Easy-Turn/Testset",
                        help="Path to evaluation dataset directory")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_audio_seconds", type=float, default=60.0)
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use: auto, cpu, cuda:0, etc.")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Loading Stage A Best Model for Evaluation")
    logger.info("=" * 60)
    logger.info(f"Base model: {args.model_path}")
    logger.info(f"Checkpoint: {args.checkpoint_path}")
    logger.info(f"Eval list: {args.eval_list_file}")
    logger.info(f"Eval dir: {args.evalset_dir}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 60)

    # 初始化模型 (Stage A, 不使用 LoRA)
    model = UAFModel(
        model_path=args.model_path,
        training_stage="A",
        use_lora=False,
    )

    # 加载最优 checkpoint 的 TD Head 权重
    td_head_path = os.path.join(args.checkpoint_path, "td_head.pt")
    if os.path.exists(td_head_path):
        logger.info(f"Loading TD Head weights from {td_head_path}")
        model.td_head.load_state_dict(torch.load(td_head_path, map_location="cpu"))
        logger.info("TD Head weights loaded successfully")
    else:
        logger.error(f"TD Head weights not found at {td_head_path}")
        return

    # 运行评估
    logger.info("\nStarting evaluation...")
    eval_metrics = evaluate_on_testset(
        model,
        args.eval_list_file,
        args.evalset_dir,
        args.max_audio_seconds,
        batch_size=args.batch_size,
    )

    # 输出结果
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    log_metrics(logger, "Testset", eval_metrics)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
