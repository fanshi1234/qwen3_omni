"""
TD Head LM - 基于 LM Head 的分类头

使用预训练的 LM Head 权重初始化，修改输出层为 4 分类。
保留原始 LM Head 用于 ASR 任务。

架构:
    输入: audio_hidden [B, A, 2048]
    输出: turn_logits [B, 4]

设计思路:
    1. 复制预训练 LM Head 的权重作为初始化
    2. 将输出层从 152064 (vocab_size) 改为 4 (num_labels)
    3. 使用 Attention Pooling 聚合序列信息
    4. 支持两种模式: pooling / last_token
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TDHeadLM(nn.Module):
    """基于 LM Head 的 TD 分类头。

    使用预训练 LM Head 的权重初始化，修改输出层为 4 分类。

    Args:
        hidden_size: 隐藏维度 (2048)
        num_labels: 分类标签数 (4)
        pooling_type: 聚合方式 ('attention' 或 'last_token')
        dropout: Dropout 概率
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_labels: int = 4,
        pooling_type: str = "attention",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.pooling_type = pooling_type

        # Attention Pooling (用于聚合序列信息)
        if pooling_type == "attention":
            self.score = nn.Linear(hidden_size, 1)

        # LayerNorm + Dropout
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        # 分类层 (输出 4 类)
        # 注意: 这里不使用预训练权重，因为输出维度不同
        self.classifier = nn.Linear(hidden_size, num_labels, bias=False)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重。"""
        if hasattr(self, 'score'):
            nn.init.normal_(self.score.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.score.bias)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)

    def load_pretrained_lm_head(self, lm_head_weight: torch.Tensor):
        """从预训练 LM Head 加载权重。

        由于输出维度不同 (152064 vs 4)，我们只加载 hidden_size 维度的权重，
        然后通过随机采样或 PCA 降维来初始化分类层。

        Args:
            lm_head_weight: 预训练 LM Head 权重 [vocab_size, hidden_size]
        """
        vocab_size, hidden_size = lm_head_weight.shape

        assert hidden_size == self.hidden_size, \
            f"hidden_size 不匹配: {hidden_size} vs {self.hidden_size}"

        # 方法 1: 随机采样 vocab_size 行中的 4 行
        # 这样可以保持预训练权重的分布特性
        indices = torch.randperm(vocab_size)[:self.num_labels]
        sampled_weight = lm_head_weight[indices]  # [4, hidden_size]

        # 复制到分类层
        with torch.no_grad():
            self.classifier.weight.copy_(sampled_weight)

        print(f"✅ 从预训练 LM Head 加载权重:")
        print(f"   原始 vocab_size: {vocab_size}")
        print(f"   采样 {self.num_labels} 行作为分类层初始化")

    def forward(
        self,
        audio_hidden: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            audio_hidden: [B, A, H] 音频编码器隐藏状态
            audio_mask: [B, A] 1=有效, 0=padding

        Returns:
            turn_logits: [B, num_labels]
        """
        B, A, H = audio_hidden.shape

        # 聚合序列信息
        if self.pooling_type == "attention":
            # Attention Pooling
            scores = self.score(audio_hidden).squeeze(-1)  # [B, A]
            if audio_mask is not None:
                scores = scores.masked_fill(audio_mask == 0, -1e4)
            weights = F.softmax(scores, dim=-1)  # [B, A]
            pooled = torch.sum(audio_hidden * weights.unsqueeze(-1), dim=1)  # [B, H]
        elif self.pooling_type == "last_token":
            # 取最后一个有效 token
            if audio_mask is not None:
                # 找到每个样本的最后一个有效位置
                last_idx = audio_mask.sum(dim=1).long() - 1  # [B]
                last_idx = last_idx.clamp(min=0)
                pooled = audio_hidden[torch.arange(B), last_idx]  # [B, H]
            else:
                pooled = audio_hidden[:, -1, :]  # [B, H]
        else:
            raise ValueError(f"不支持的 pooling_type: {self.pooling_type}")

        # Norm + Dropout + 分类
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # [B, num_labels]

        return logits


class DualLMHead(nn.Module):
    """双 LM Head 架构。

    包含两个 LM Head:
    1. ASR Head: 保持原始预训练权重，用于语音识别
    2. TD Head: 基于预训练权重，修改为 4 分类

    Args:
        hidden_size: 隐藏维度 (2048)
        vocab_size: 词表大小 (152064)
        num_labels: 分类标签数 (4)
        pooling_type: TD Head 的聚合方式
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        vocab_size: int = 152064,
        num_labels: int = 4,
        pooling_type: str = "attention",
    ):
        super().__init__()

        # ASR Head (保持原始预训练权重)
        self.asr_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # TD Head (基于预训练权重，修改为 4 分类)
        self.td_head = TDHeadLM(
            hidden_size=hidden_size,
            num_labels=num_labels,
            pooling_type=pooling_type,
        )

    def load_pretrained_weights(self, lm_head_weight: torch.Tensor):
        """从预训练 LM Head 加载权重。

        Args:
            lm_head_weight: 预训练 LM Head 权重 [vocab_size, hidden_size]
        """
        # 1. 加载 ASR Head (完全复制)
        with torch.no_grad():
            self.asr_head.weight.copy_(lm_head_weight)

        # 2. 加载 TD Head (采样初始化)
        self.td_head.load_pretrained_lm_head(lm_head_weight)

        print(f"✅ 双 LM Head 权重加载完成")

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
        mode: str = "both",
    ) -> dict[str, torch.Tensor]:
        """前向传播。

        Args:
            hidden_states: [B, A, H] 音频编码器隐藏状态
            audio_mask: [B, A] 1=有效, 0=padding
            mode: 模式 ('asr', 'td', 'both')

        Returns:
            dict: 包含 'lm_logits' 和/或 'turn_logits'
        """
        result = {}

        if mode in ["asr", "both"]:
            # ASR: 对每个位置预测词表
            lm_logits = self.asr_head(hidden_states)  # [B, A, vocab_size]
            result["lm_logits"] = lm_logits

        if mode in ["td", "both"]:
            # TD: 聚合后分类
            turn_logits = self.td_head(hidden_states, audio_mask)  # [B, num_labels]
            result["turn_logits"] = turn_logits

        return result


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    # 测试 TDHeadLM
    print("测试 TDHeadLM:")
    head = TDHeadLM(hidden_size=2048, num_labels=4, pooling_type="attention")
    x = torch.randn(2, 100, 2048)
    mask = torch.ones(2, 100)
    out = head(x, mask)
    print(f"  输入: {x.shape}")
    print(f"  输出: {out.shape}")
    assert out.shape == (2, 4), f"形状错误: {out.shape}"

    # 测试 load_pretrained_lm_head
    fake_lm_weight = torch.randn(152064, 2048)
    head.load_pretrained_lm_head(fake_lm_weight)
    print(f"  预训练权重加载成功")

    # 测试 DualLMHead
    print("\n测试 DualLMHead:")
    dual_head = DualLMHead(
        hidden_size=2048,
        vocab_size=152064,
        num_labels=4,
        pooling_type="attention",
    )

    # 加载预训练权重
    dual_head.load_pretrained_weights(fake_lm_weight)

    # 前向传播
    result = dual_head(x, mask, mode="both")
    print(f"  lm_logits shape: {result['lm_logits'].shape}")
    print(f"  turn_logits shape: {result['turn_logits'].shape}")
    assert result['lm_logits'].shape == (2, 100, 152064)
    assert result['turn_logits'].shape == (2, 4)

    print("\n✅ 所有测试通过!")
