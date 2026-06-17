#!/usr/bin/env python3
"""
批量更新 .list 文件中的 duration 字段
使用多进程并行读取 wav 文件头部
"""

import os
import json
import wave
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def get_wav_duration(wav_path: str) -> float:
    """快速获取 wav 文件时长（只读头部）"""
    try:
        with wave.open(wav_path, 'r') as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0


def process_list_file(args: tuple) -> dict:
    """处理单个 .list 文件，补充 duration"""
    list_file, trainset_dir = args

    # 读取所有条目
    entries = []
    with open(list_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    # 更新 duration
    updated = 0
    for entry in entries:
        wav_path = os.path.join(trainset_dir, entry['wav'].lstrip('./'))
        if os.path.exists(wav_path):
            duration = get_wav_duration(wav_path)
            entry['duration'] = round(duration, 3)
            updated += 1

    # 写回文件
    with open(list_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    return {
        'file': os.path.basename(list_file),
        'total': len(entries),
        'updated': updated
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list_dir', default='/data1/wgy/qwen/dataset/Easy-Turn/Trainset_list')
    parser.add_argument('--trainset_dir', default='/data1/wgy/qwen/dataset/Easy-Turn/Trainset')
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()

    # 收集所有 .list 文件
    list_files = [
        os.path.join(args.list_dir, f)
        for f in os.listdir(args.list_dir)
        if f.endswith('.list')
    ]

    logger.info(f"=" * 60)
    logger.info(f"Updating duration for {len(list_files)} files")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"=" * 60)

    total_entries = 0
    total_updated = 0
    start_time = time.time()

    # 并行处理
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        tasks = [(f, args.trainset_dir) for f in list_files]

        with tqdm(total=len(tasks), desc="Processing files", unit="file") as pbar:
            futures = {executor.submit(process_list_file, t): t for t in tasks}

            for future in as_completed(futures):
                result = future.result()
                total_entries += result['total']
                total_updated += result['updated']
                pbar.update(1)
                pbar.set_postfix({
                    'file': result['file'][:20],
                    'entries': f"{result['updated']}/{result['total']}"
                })

    elapsed = time.time() - start_time

    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Duration update completed!")
    logger.info(f"   Total entries: {total_entries}")
    logger.info(f"   Updated: {total_updated}")
    logger.info(f"   Time: {elapsed:.1f}s")
    logger.info(f"   Speed: {total_updated/elapsed:.0f} files/s")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
