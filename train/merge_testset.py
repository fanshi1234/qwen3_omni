#!/usr/bin/env python3
"""
合并 Testset 的 4 个类别为一个评估集
"""

import json
from pathlib import Path

TESTSET_DIR = Path("./dataset/Easy-Turn/Testset")
OUTPUT_FILE = TESTSET_DIR / "testset_all.list"

# 4 个类别的 list 文件
LIST_FILES = [
    "complete/complete_test.list",
    "incomplete/incomplete_real_test.list",
    "backchannel/backchannel_test.list",
    "wait/wait_test.list",
]

def main():
    all_data = []
    stats = {}

    print("合并 Testset:")
    print("-" * 50)

    for rel_path in LIST_FILES:
        fpath = TESTSET_DIR / rel_path
        if not fpath.exists():
            print(f"⚠️  跳过 {rel_path} (文件不存在)")
            continue

        count = 0
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    all_data.append(item)
                    count += 1

        # 统计类别
        category = rel_path.split('/')[0]
        stats[category] = count
        print(f"✅ {rel_path}: {count} 样本")

    # 保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("-" * 50)
    print(f"✅ 合并完成!")
    print(f"   总样本数: {len(all_data)}")
    print(f"   输出文件: {OUTPUT_FILE}")
    print()
    print("类别分布:")
    for cat, count in stats.items():
        print(f"   {cat}: {count}")

if __name__ == "__main__":
    main()
