#!/usr/bin/env python3
"""
快速转换 Easy-Turn 训练集为 JSONL 格式
带进度条显示
"""

import os
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional
import logging
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_sample(prefix_path: str, dataset_name: str) -> Optional[Dict]:
    """处理单个样本"""
    try:
        fields = {}
        for ext in ['task', 'txt', 'lang', 'state', 'speaker', 'emotion', 'gender', 'extra']:
            file_path = f"{prefix_path}.{ext}"
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    fields[ext] = f.read().strip()
            else:
                fields[ext] = ""

        wav_path = f"{prefix_path}.wav"
        if not os.path.exists(wav_path):
            return None

        key = os.path.basename(prefix_path)

        extra = {}
        if fields['extra']:
            try:
                extra = json.loads(fields['extra'])
            except:
                extra = {"dataset": dataset_name}

        rel_wav_path = os.path.relpath(wav_path, '/data1/wgy/qwen/dataset/Easy-Turn/Trainset')

        return {
            "task": fields['task'] or "<TRANSCRIBE> <BACKCHANNEL> <COMPLETE>",
            "key": key,
            "wav": f"./{rel_wav_path}",
            "txt": fields['txt'],
            "lang": fields['lang'] or "<CN>",
            "speaker": fields['speaker'] if fields['speaker'] and fields['speaker'] != '<NONE>' else "UNKNOWN",
            "emotion": fields['emotion'] if fields['emotion'] and fields['emotion'] != '<NONE>' else "<NONE>",
            "gender": fields['gender'] if fields['gender'] and fields['gender'] != '<NONE>' else "UNKNOWN",
            "duration": 0.0,
            "state": fields['state'] or "0",
            "extra": extra
        }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/data1/wgy/qwen/dataset/Easy-Turn/Trainset')
    parser.add_argument('--output_dir', default='/data1/wgy/qwen/dataset/Easy-Turn/Trainset_list')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    datasets = [
        ('complete_syn', 'complete'),
        ('incomplete_syn', 'incomplete'),
        ('magicdata_ramc', 'mixed'),
        ('renzao_8.13', 'mixed'),
        ('8.13', 'mixed'),
    ]

    total_written = 0
    start_time = time.time()

    for dataset_name, category in datasets:
        data_dir = os.path.join(args.input_dir, dataset_name)
        if not os.path.exists(data_dir):
            logger.info(f"⏭️  Skipping {dataset_name} (not found)")
            continue

        # 收集所有 .task 文件
        task_files = list(Path(data_dir).rglob("*.task"))
        logger.info(f"\n📂 {dataset_name}: {len(task_files)} samples")

        # 处理样本
        results = []
        for task_file in tqdm(task_files, desc=f"  Processing", unit="files", ncols=80):
            prefix = str(task_file).replace('.task', '')
            result = process_sample(prefix, dataset_name)
            if result:
                results.append(result)

        # 写入文件
        output_file = os.path.join(args.output_dir, f"{dataset_name}.list")
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in results:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        total_written += len(results)
        logger.info(f"  ✅ Written {len(results)} entries")

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"🎉 Conversion completed!")
    logger.info(f"   Total: {total_written} samples")
    logger.info(f"   Time: {elapsed:.1f}s")
    logger.info(f"   Speed: {total_written/elapsed:.0f} samples/s")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
