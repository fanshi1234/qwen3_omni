"""
损失函数工具库

当前 train.py 直接使用 F.cross_entropy + CLASS_WEIGHTS
本文件提供更高级的损失函数，供后续优化使用
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class AdaptiveWeightLoss(nn.Module):
    """
    自适应权重损失函数

    支持多种权重策略:
    - sqrt: sqrt 逆频率 (推荐)
    - ens: Effective Number of Samples
    - inv: 经典逆频率
    - none: 不使用权重
    """

    def __init__(
        self,
        num_classes: int = 4,
        strategy: str = "sqrt",
        label_counts: Optional[List[int]] = None,
        max_weight: float = 4.0,
        beta: float = 0.9999,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strategy = strategy

        if label_counts is not None and strategy != "none":
            weights = self._compute_weights(label_counts, strategy, max_weight, beta)
        else:
            weights = [1.0] * num_classes

        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        self.ce_loss = nn.CrossEntropyLoss(
            weight=self.weights,
            reduction='mean',
            label_smoothing=label_smoothing,
        )

    def _compute_weights(self, label_counts, strategy, max_weight, beta):
        if strategy == "sqrt":
            return self._sqrt_weights(label_counts, max_weight)
        elif strategy == "ens":
            return self._ens_weights(label_counts, beta)
        elif strategy == "inv":
            return self._inv_weights(label_counts, max_weight)
        else:
            return [1.0] * len(label_counts)

    def _sqrt_weights(self, label_counts, max_weight):
        max_count = max(label_counts)
        return [min(math.sqrt(max_count / c), max_weight) for c in label_counts]

    def _ens_weights(self, label_counts, beta):
        effective_num = [1.0 - beta**c for c in label_counts]
        weights = [(1.0 - beta) / e for e in effective_num]
        total = sum(weights)
        return [w / total * len(label_counts) for w in weights]

    def _inv_weights(self, label_counts, max_weight):
        total = sum(label_counts)
        return [min(total / (len(label_counts) * c), max_weight) for c in label_counts]

    def forward(self, logits, labels):
        return self.ce_loss(logits, labels)


class FocalLoss(nn.Module):
    """
    Focal Loss (ICCV 2017)

    适用于难易样本不均衡的场景
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer('alpha_tensor', torch.tensor(alpha, dtype=torch.float32))

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha = self.alpha_tensor.to(inputs.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss (CVPR 2019)

    结合 Effective Number of Samples + Focal Loss
    """

    def __init__(self, label_counts, beta=0.9999, gamma=2.0):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=gamma, reduction='none')

        effective_num = [1.0 - beta**c for c in label_counts]
        weights = [(1.0 - beta) / e for e in effective_num]
        total = sum(weights)
        weights = [w / total * len(label_counts) for w in weights]

        self.register_buffer('alpha', torch.tensor(weights, dtype=torch.float32))

    def forward(self, inputs, targets):
        fl = self.focal_loss(inputs, targets)
        alpha = self.alpha.to(inputs.device)
        alpha_t = alpha[targets]
        return (alpha_t * fl).mean()


def get_loss_function(strategy="sqrt", label_counts=None, num_classes=4):
    """获取损失函数"""
    if strategy == "focal":
        return FocalLoss(gamma=2.0)
    elif strategy == "cb":
        if label_counts is None:
            raise ValueError("label_counts is required for CB Loss")
        return ClassBalancedLoss(label_counts=label_counts)
    else:
        return AdaptiveWeightLoss(
            num_classes=num_classes,
            strategy=strategy,
            label_counts=label_counts,
        )


if __name__ == "__main__":
    # 测试
    batch_size, num_classes = 4, 4
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    label_counts = [423678, 712404, 41724, 40061]

    for strategy in ["sqrt", "ens", "inv", "focal", "cb"]:
        loss_fn = get_loss_function(strategy=strategy, label_counts=label_counts)
        loss = loss_fn(logits, labels)
        print(f"{strategy}: {loss.item():.4f}")
