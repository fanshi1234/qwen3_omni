"""
TD Head MLP - 两层 MLP 版本

输入: audio_hidden [B, A, H]
输出: turn_logits [B, 4]
"""

import torch
import torch.nn as nn


class TDHeadMLP(nn.Module):
    def __init__(self, hidden_size: int = 2048, num_labels: int = 4, dropout: float = 0.1):
        super().__init__()
        # Attention Pooling
        self.score = nn.Linear(hidden_size, 1)

        # 两层 MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )

        # 初始化
        nn.init.normal_(self.score.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.score.bias)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                nn.init.zeros_(layer.bias)

    def forward(self, audio_hidden, audio_mask=None):
        """
        audio_hidden: [B, A, H]
        audio_mask: [B, A]  1=有效, 0=padding
        """
        scores = self.score(audio_hidden).squeeze(-1)  # [B, A]
        if audio_mask is not None:
            scores = scores.masked_fill(audio_mask == 0, -1e4)
        weights = torch.softmax(scores, dim=-1)  # [B, A]
        pooled = torch.sum(audio_hidden * weights.unsqueeze(-1), dim=1)  # [B, H]
        return self.mlp(pooled)  # [B, 4]
