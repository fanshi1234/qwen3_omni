#!/usr/bin/env python3
"""
合并多个数据集，平衡类别分布
"""

import json
import random
from pathlib import Path
from collections import Counter

# 配置
TRAINSET_LIST_DIR = Path("./dataset/Easy-Turn/Trainset_list")
OUTPUT_FILE = TRAINSET_LIST_DIR / "merged_balanced.list"

# 数据集配置 (文件名, 采样数量)
DATASETS = [
    ("magicdata_ramc.list", None),      # 全部使用 (143k, 有 C/I/B)
    ("8.13.list", None),                 # 全部使用 (40k, 只有 W)
    ("renzao_8.13.list", None),          # 全部使用 (21k, 只有 B)
    ("complete_syn.list", 50000),        # 采样 50k (平衡 Complete)
    ("incomplete_syn.list", 50000),      # 采样 50k (平衡 InComplete)
]

def load_dataset(list_file, max_samples=None):
    """加载数据集"""
    data = []
    with open(list_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    if max_samples and len(data) > max_samples:
        data = random.sample(data, max_samples)

    return data

def get_label(txt):
    """获取标签"""
    if txt.endswith("<COMPLETE>"):
        return 0
    elif txt.endswith("<INCOMPLETE>"):
        return 1
    elif txt.endswith("<BACKCHANNEL>"):
        return 2
    elif txt.endswith("<WAIT>"):
        return 3
    return -1

def main():
    random.seed(42)

    all_data = []
    label_counts = Counter()

    print("合并数据集:")
    print("-" * 50)

    for fname, max_samples in DATASETS:
        fpath = TRAINSET_LIST_DIR / fname
        if not fpath.exists():
            print(f"⚠️  跳过 {fname} (文件不存在)")
            continue

        data = load_dataset(fpath, max_samples)

        # 统计类别
        counts = Counter()
        for item in data:
            label = get_label(item.get('txt', ''))
            counts[label] += 1

        all_data.extend(data)
        label_counts += counts

        print(f"✅ {fname}: {len(data)} 样本")
        print(f"   类别: {dict(counts)}")

    # 打乱顺序
    random.shuffle(all_data)

    # 保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("-" * 50)
    print(f"✅ 合并完成!")
    print(f"   总样本数: {len(all_data)}")
    print(f"   输出文件: {OUTPUT_FILE}")
    print()
    print("合并后类别分布:")
    label_names = {0: "Complete", 1: "InComplete", 2: "Backchannel", 3: "Wait"}
    for label_id in sorted(label_counts.keys()):
        count = label_counts[label_id]
        pct = count / len(all_data) * 100
        print(f"   {label_names[label_id]:>12}: {count:>8} ({pct:.1f}%)")

if __name__ == "__main__":
    main()
