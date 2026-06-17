#!/usr/bin/env python3
"""
将 Easy-Turn 训练集转换为与测试集相同的 JSONL 格式
充分利用多进程加速: 96核 CPU + 503GB 内存
"""

import os
import json
import argparse
import time
import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_audio_duration(wav_path: str) -> float:
    """获取音频时长，失败返回 0"""
    try:
        # 使用 wave 模块 (标准库，无需安装)
        import wave
        with wave.open(wav_path, 'r') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / rate
    except Exception:
        try:
            # 备选: 使用 ffprobe
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', wav_path],
                capture_output=True, text=True, timeout=5
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0


def process_sample(sample_info: tuple) -> Optional[Dict]:
    """
    处理单个样本，将分离文件合并为 JSONL 格式

    Args:
        sample_info: (prefix_path, category, dataset_name)

    Returns:
        dict: JSONL 条目，或 None 如果处理失败
    """
    prefix_path, category, dataset_name = sample_info

    try:
        # 读取所有字段
        fields = {}
        for ext in ['task', 'txt', 'lang', 'state', 'speaker', 'emotion', 'gender', 'duration', 'extra']:
            file_path = f"{prefix_path}.{ext}"
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    fields[ext] = f.read().strip()
            else:
                fields[ext] = ""

        # 构建 wav 路径
        wav_path = f"{prefix_path}.wav"
        if not os.path.exists(wav_path):
            return None

        # 获取实际时长
        duration = get_audio_duration(wav_path)

        # 生成 key (使用文件名)
        key = os.path.basename(prefix_path)

        # 解析 extra
        extra = {}
        if fields['extra']:
            try:
                extra = json.loads(fields['extra'])
            except json.JSONDecodeError:
                extra = {"dataset": dataset_name}

        # 构建相对路径 (相对于 Trainset 目录)
        rel_wav_path = os.path.relpath(wav_path, os.path.dirname(os.path.dirname(prefix_path)))

        # 构建 JSONL 条目
        entry = {
            "task": fields['task'] or "<TRANSCRIBE> <BACKCHANNEL> <COMPLETE>",
            "key": key,
            "wav": f"./{rel_wav_path}",
            "txt": fields['txt'],
            "lang": fields['lang'] or "<CN>",
            "speaker": fields['speaker'] if fields['speaker'] and fields['speaker'] != '<NONE>' else "UNKNOWN",
            "emotion": fields['emotion'] if fields['emotion'] and fields['emotion'] != '<NONE>' else "<NONE>",
            "gender": fields['gender'] if fields['gender'] and fields['gender'] != '<NONE>' else "UNKNOWN",
            "duration": duration,
            "state": fields['state'] or "0",
            "extra": extra
        }

        return entry

    except Exception as e:
        logger.debug(f"Failed to process {prefix_path}: {e}")
        return None


def collect_samples(data_dir: str, category: str, dataset_name: str) -> List[tuple]:
    """收集目录中所有样本的前缀路径"""
    samples = []
    data_path = Path(data_dir)

    # 查找所有 .task 文件
    for task_file in data_path.rglob("*.task"):
        prefix = str(task_file).replace('.task', '')
        samples.append((prefix, category, dataset_name))

    return samples


def process_dataset(args: tuple) -> tuple:
    """处理单个数据集目录"""
    data_dir, category, dataset_name, output_dir = args

    logger.info(f"Processing: {category}/{dataset_name} from {data_dir}")

    if not os.path.exists(data_dir):
        logger.warning(f"Directory not found: {data_dir}")
        return category, dataset_name, 0

    # 收集样本
    samples = collect_samples(data_dir, category, dataset_name)
    logger.info(f"  Found {len(samples)} samples")

    if not samples:
        return category, dataset_name, 0

    # 多进程处理
    results = []
    with ProcessPoolExecutor(max_workers=min(32, os.cpu_count() // 2)) as executor:
        futures = {executor.submit(process_sample, s): s for s in samples}

        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    # 写入输出文件
    output_file = os.path.join(output_dir, f"{dataset_name}.list")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    logger.info(f"  Written {len(results)} entries to {output_file}")
    return category, dataset_name, len(results)


def main():
    parser = argparse.ArgumentParser(description='Convert Easy-Turn training set to JSONL format')
    parser.add_argument('--input_dir', type=str,
                        default='/data1/wgy/qwen/dataset/Easy-Turn/Trainset',
                        help='Input training set directory')
    parser.add_argument('--output_dir', type=str,
                        default='/data1/wgy/qwen/dataset/Easy-Turn/Trainset_list',
                        help='Output directory for .list files')
    parser.add_argument('--max_workers', type=int, default=16,
                        help='Max parallel dataset processing workers')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 定义数据集映射
    # 格式: (目录名, 类别, 输出文件名)
    datasets = [
        # complete_syn
        (os.path.join(args.input_dir, 'complete_syn'), 'complete', 'complete_syn'),
        # incomplete_syn
        (os.path.join(args.input_dir, 'incomplete_syn'), 'incomplete', 'incomplete_syn'),
        # magicdata_ramc
        (os.path.join(args.input_dir, 'magicdata_ramc'), 'mixed', 'magicdata_ramc'),
        # renzao_8.13
        (os.path.join(args.input_dir, 'renzao_8.13'), 'mixed', 'renzao_8_13'),
        # 8.13
        (os.path.join(args.input_dir, '8.13'), 'mixed', 'wait_8_13'),
    ]

    # 处理 tar.gz 文件 (需要先解压)
    tar_datasets = [
        ('backchannel_real.tar.gz', 'backchannel', 'backchannel_real'),
        ('backchannel_syn.tar.gz', 'backchannel', 'backchannel_syn'),
        ('complete_real.tar.gz', 'complete', 'complete_real'),
        ('incomplete_real.tar.gz', 'incomplete', 'incomplete_real'),
        ('wait.tar.gz', 'wait', 'wait'),
    ]

    # 检查已解压的目录
    for tar_name, category, dataset_name in tar_datasets:
        # 检查是否已解压
        extracted_dir = os.path.join(args.input_dir, dataset_name.replace('_real', '').replace('_syn', ''))
        if not os.path.exists(extracted_dir):
            # 尝试其他可能的目录名
            alt_dir = os.path.join(args.input_dir, tar_name.replace('.tar.gz', ''))
            if os.path.exists(alt_dir):
                extracted_dir = alt_dir
            else:
                logger.info(f"Skipping {tar_name} (not extracted)")
                continue

        if os.path.isdir(extracted_dir):
            datasets.append((extracted_dir, category, dataset_name))

    logger.info(f"=" * 60)
    logger.info(f"Processing {len(datasets)} datasets")
    logger.info(f"Input: {args.input_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Workers: {args.max_workers}")
    logger.info(f"=" * 60)

    # 并行处理数据集
    total_samples = 0
    start_time = time.time()

    # 使用进程池并行处理多个数据集
    dataset_args = [(d[0], d[1], d[2], args.output_dir) for d in datasets if os.path.exists(d[0])]

    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(process_dataset, arg): arg for arg in dataset_args}

        for future in as_completed(futures):
            category, dataset_name, count = future.result()
            total_samples += count
            logger.info(f"✅ {category}/{dataset_name}: {count} samples")

    elapsed = time.time() - start_time

    # 统计结果
    logger.info(f"=" * 60)
    logger.info(f"Conversion completed!")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Time elapsed: {elapsed:.1f} seconds")
    logger.info(f"Speed: {total_samples/elapsed:.0f} samples/sec")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"=" * 60)

    # 列出生成的文件
    logger.info("\nGenerated files:")
    for f in sorted(os.listdir(args.output_dir)):
        if f.endswith('.list'):
            fpath = os.path.join(args.output_dir, f)
            with open(fpath) as fp:
                lines = len(fp.readlines())
            logger.info(f"  {f}: {lines} entries")


if __name__ == "__main__":
    main()
