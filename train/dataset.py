"""
Easy-Turn 数据集
"""

import json
import wave
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Optional
from collections import Counter

# 尝试导入 librosa（用于重采样）
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("⚠️  librosa not installed, audio resampling disabled")

from config import LABEL2ID


class TurnDataset(Dataset):
    def __init__(
        self,
        list_file: str,
        trainset_dir: str,
        max_duration: float = 10.0,
        max_samples: Optional[int] = None,
    ):
        self.data = []
        self.max_duration = max_duration
        self.trainset_dir = Path(trainset_dir)
        self.label_counts = Counter()

        with open(list_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                txt = item.get('txt', '')
                label, clean_text = self._extract(txt)
                if label < 0:
                    continue
                if item.get('duration', 0) > max_duration:
                    continue
                item['turn_label'] = label
                item['text_target'] = clean_text
                self.data.append(item)
                self.label_counts[label] += 1
                if max_samples and len(self.data) >= max_samples:
                    break

        print(f"✅ Loaded {len(self.data)} samples from {list_file}")
        print(f"   类别分布: {dict(self.label_counts)}")

    def _extract(self, txt):
        """提取标签和纯文本"""
        for tag, lid in LABEL2ID.items():
            if txt.endswith(tag):
                return lid, txt[:-len(tag)].strip()
        return -1, ""

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        wav_path = str(self.trainset_dir / item['wav'].lstrip('./'))

        # 加载音频
        try:
            with wave.open(wav_path, 'r') as wf:
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                sr = wf.getframerate()
        except Exception:
            audio = np.zeros(16000, dtype=np.float32)
            sr = 16000

        # 重采样到 16kHz
        if sr != 16000 and HAS_LIBROSA:
            try:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            except Exception:
                pass

        # 截断到最大长度
        max_samples = int(self.max_duration * 16000)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        return {
            "audio_path": wav_path,
            "turn_label": item['turn_label'],
            "text_target": item['text_target'],
            "key": item.get('key', ''),
        }


def collate_fn(batch):
    """Collate 函数"""
    return {
        "audio_list": [x['audio_path'] for x in batch],
        "turn_labels": torch.tensor([x['turn_label'] for x in batch], dtype=torch.long),
        "text_targets": [x['text_target'] for x in batch],
        "keys": [x['key'] for x in batch],
    }


def get_weighted_sampler(dataset):
    """
    创建加权采样器，平衡类别分布

    使每个类别被采样的概率相等
    """
    label_counts = dataset.label_counts
    total = sum(label_counts.values())
    num_classes = len(label_counts)

    # 类别权重
    class_weights = {}
    for label_id, count in label_counts.items():
        class_weights[label_id] = total / (num_classes * count)

    # 每个样本的权重
    sample_weights = []
    for item in dataset.data:
        label = item['turn_label']
        sample_weights.append(class_weights[label])

    print(f"📊 WeightedRandomSampler 配置:")
    print(f"   类别权重: {class_weights}")

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,
    )

    return sampler


def get_balanced_sampler(dataset, samples_per_class=None):
    """
    创建平衡采样器，每个类别采样相同数量
    """
    # 按类别分组
    class_indices = {0: [], 1: [], 2: [], 3: []}
    for idx, item in enumerate(dataset.data):
        class_indices[item['turn_label']].append(idx)

    # 确定每个类别的采样数量
    if samples_per_class is None:
        samples_per_class = min(len(indices) for indices in class_indices.values())

    print(f"📊 BalancedSampler 配置:")
    print(f"   每类采样: {samples_per_class}")

    # 生成平衡的索引列表
    balanced_indices = []
    for label_id, indices in class_indices.items():
        if len(indices) >= samples_per_class:
            selected = np.random.choice(indices, samples_per_class, replace=False)
        else:
            selected = np.random.choice(indices, samples_per_class, replace=True)
        balanced_indices.extend(selected)

    np.random.shuffle(balanced_indices)

    sampler = torch.utils.data.SubsetRandomSampler(balanced_indices)

    return sampler
