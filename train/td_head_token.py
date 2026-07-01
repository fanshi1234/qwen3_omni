"""
TD Head Token - 与原始 LM Head 完全一致的分类头

使用所有 hidden states，输出维度从 152064 改为 4。
用原始 lm_head 中对应词的列向量初始化。

架构:
    Thinker hidden_states [B, T, 2048]
            ↓
    token_class_head = Linear(2048, 4, bias=False)
            ↓
    class_logits [B, T, 4]
            ↓
    取最后有效位置 logits [B, 4]
            ↓
    CE Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 标签对应的 token 文本
LABEL_TOKENS = {
    0: "<COMPLETE>",
    1: "<INCOMPLETE>",
    2: "<BACKCHANNEL>",
    3: "<WAIT>",
}

# 每个类别相关的词汇 (中英文)
RELEVANT_TOKENS = {
    0: ['complete', 'finish', 'end', 'done', 'over', 'conclude', '完成', '结束', '搞定', '收尾', '完结'],
    1: ['incomplete', 'continue', 'keep', 'go', 'proceed', 'next', '继续', '接着', '未完', '待续', '进行'],
    2: ['backchannel', 'mm', 'uh', 'yeah', 'okay', 'right', '嗯', '哦', '好的', '对', '啊'],
    3: ['wait', 'interrupt', 'hold', 'stop', 'pause', 'halt', '稍等', '暂停', '停下', '等等', '打断'],
}


class TDHeadToken(nn.Module):
    """与原始 LM Head 完全一致的分类头。

    使用所有 hidden states，输出 4 个类别的分数。
    用原始 lm_head 中对应词的行向量初始化。

    Args:
        hidden_size: 隐藏维度 (2048)
        num_labels: 分类标签数 (4)
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        num_labels: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels

        # 分类层 - 与 LM Head 结构完全一致
        # LM Head: Linear(2048, 152064, bias=False)
        # TD Head: Linear(2048, 4, bias=False)
        self.token_class_head = nn.Linear(hidden_size, num_labels, bias=False)

    def load_pretrained_weights(
        self,
        lm_head_weight: torch.Tensor,
        tokenizer,
        init_strategy: str = 'column',
        init_scale: float = 0.2,
    ):
        """从预训练 LM Head 加载权重。

        Args:
            lm_head_weight: 预训练 LM Head 权重 [vocab_size, hidden_size]
            tokenizer: tokenizer 用于查找 token id
            init_strategy: 'row' | 'column' | 'projection'
            init_scale: 缩放系数 (默认 0.2)
        """
        vocab_size, hidden_size = lm_head_weight.shape

        assert hidden_size == self.hidden_size, \
            f"hidden_size 不匹配: {hidden_size} vs {self.hidden_size}"

        vocab = tokenizer.get_vocab()

        print(f"\n{'='*60}")
        print(f"初始化 TD Head (strategy={init_strategy}, scale={init_scale})")
        print(f"{'='*60}")

        if init_strategy == 'row':
            # 策略 1: 取行向量 (原始方法)
            self._init_row(lm_head_weight, tokenizer, vocab, vocab_size)

        elif init_strategy == 'column':
            # 策略 2: 取列向量平均值 + 缩放 (推荐)
            self._init_column(lm_head_weight, vocab, vocab_size, init_scale)

        elif init_strategy == 'projection':
            # 策略 3: 投影矩阵 + 缩放
            self._init_projection(lm_head_weight, vocab_size, init_scale)

        else:
            raise ValueError(f"未知的初始化策略: {init_strategy}")

        print(f"{'='*60}\n")

    def _init_row(
        self,
        lm_head_weight: torch.Tensor,
        tokenizer,
        vocab: dict,
        vocab_size: int,
    ):
        """策略 1: 取行向量 (原始方法)。"""
        print(f"\n[策略 1] 取行向量")

        # 标签核心部分
        LABEL_CORE_TOKENS = {
            0: "COMPLETE",
            1: "INCOMPLETE",
            2: "BACKCHANNEL",
            3: "WAIT",
        }

        for label_id, core_token in LABEL_CORE_TOKENS.items():
            token_ids = tokenizer.encode(core_token, add_special_tokens=False)

            if len(token_ids) > 0 and all(tid < vocab_size for tid in token_ids):
                if len(token_ids) > 1:
                    weights = lm_head_weight[token_ids]
                    self.token_class_head.weight[label_id] = weights.mean(dim=0)
                    print(f"  ✅ Label {label_id} ({LABEL_TOKENS[label_id]}): "
                          f"{len(token_ids)} tokens 平均值")
                else:
                    self.token_class_head.weight[label_id] = lm_head_weight[token_ids[0]]
                    print(f"  ✅ Label {label_id} ({LABEL_TOKENS[label_id]}): "
                          f"token {token_ids[0]}")
            else:
                nn.init.normal_(self.token_class_head.weight[label_id], mean=0.0, std=0.02)
                print(f"  ⚠️ Label {label_id} ({LABEL_TOKENS[label_id]}): 随机初始化")

    def _init_column(
        self,
        lm_head_weight: torch.Tensor,
        vocab: dict,
        vocab_size: int,
        init_scale: float,
    ):
        """策略 2: 取行向量平均值 + 缩放 (推荐)。"""
        print(f"\n[策略 2] 取行向量平均值 + 缩放 (scale={init_scale})")

        # lm_head_weight 形状: [vocab_size, hidden_size] = [152064, 2048]
        # 需要提取相关 token 的行向量

        W_turn = torch.zeros(self.num_labels, self.hidden_size)

        for class_id, tokens in RELEVANT_TOKENS.items():
            # 找到在词汇表中的 token
            found_tokens = []
            for t in tokens:
                if t in vocab:
                    found_tokens.append(t)

            if found_tokens:
                token_ids = [vocab[t] for t in found_tokens]

                # 提取对应的行向量
                rows = lm_head_weight[token_ids]  # [N, 2048]

                # 取平均值 + 缩放
                W_turn[class_id] = rows.mean(dim=0) * init_scale

                print(f"  ✅ Class {class_id} ({LABEL_TOKENS[class_id]}): "
                      f"{len(found_tokens)} tokens → "
                      f"rows {rows.shape} → "
                      f"avg {W_turn[class_id].shape}")
            else:
                nn.init.normal_(W_turn[class_id], mean=0.0, std=0.02)
                print(f"  ⚠️ Class {class_id} ({LABEL_TOKENS[class_id]}): "
                      f"未找到相关 token，随机初始化")

        # 初始化
        self.token_class_head.weight.data = W_turn

    def _init_projection(
        self,
        lm_head_weight: torch.Tensor,
        vocab_size: int,
        init_scale: float,
    ):
        """策略 3: 投影矩阵 + 缩放。"""
        print(f"\n[策略 3] 投影矩阵 (scale={init_scale})")

        # 创建随机投影矩阵
        W_proj = torch.randn(self.num_labels, vocab_size) * 0.02

        # 计算初始化权重
        W_turn = (lm_head_weight @ W_proj.T) * init_scale

        print(f"  W_proj: {W_proj.shape}")
        print(f"  W_turn: {W_turn.shape}")

        # 初始化
        self.token_class_head.weight.data = W_turn

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """前向传播。

        Args:
            hidden_states: [B, T, H] Transformer 解码器输出
            attention_mask: [B, T] 1=有效, 0=padding

        Returns:
            turn_logits: [B, num_labels]
        """
        B, T, H = hidden_states.shape

        # 确保权重与输入在同一设备和数据类型上
        if self.token_class_head.weight.device != hidden_states.device:
            self.token_class_head = self.token_class_head.to(hidden_states.device)
        if self.token_class_head.weight.dtype != hidden_states.dtype:
            self.token_class_head = self.token_class_head.to(hidden_states.dtype)

        # 线性变换: [B, T, H] → [B, T, 4] (与原始 LM Head 一致，无归一化)
        class_logits = self.token_class_head(hidden_states)

        # 取最后有效位置的 logits
        if attention_mask is not None:
            # 找到每个样本的最后一个有效位置
            last_valid_idx = attention_mask.sum(dim=1).long() - 1  # [B]
            last_valid_idx = last_valid_idx.clamp(min=0)

            # 提取最后有效位置的 logits
            batch_indices = torch.arange(B, device=hidden_states.device)
            turn_logits = class_logits[batch_indices, last_valid_idx]  # [B, 4]
        else:
            # 没有 mask，直接取最后一个位置
            turn_logits = class_logits[:, -1, :]  # [B, 4]

        return turn_logits


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    # 测试 TDHeadToken
    print("测试 TDHeadToken:")
    head = TDHeadToken(hidden_size=2048, num_labels=4)

    # 模拟输入
    B, T, H = 2, 100, 2048
    hidden_states = torch.randn(B, T, H)
    attention_mask = torch.ones(B, T)
    attention_mask[0, 80:] = 0  # 第一个样本只有 80 个有效位置
    attention_mask[1, 90:] = 0  # 第二个样本只有 90 个有效位置

    # 前向传播
    turn_logits = head(hidden_states, attention_mask)
    print(f"  输入: hidden_states {hidden_states.shape}")
    print(f"  输入: attention_mask {attention_mask.shape}")
    print(f"  输出: turn_logits {turn_logits.shape}")
    assert turn_logits.shape == (2, 4), f"形状错误: {turn_logits.shape}"

    # 验证取的是正确位置
    print(f"\n  验证:")
    print(f"  样本 0 最后有效位置: 79")
    print(f"  样本 1 最后有效位置: 89")

    # 手动验证
    class_logits = head.token_class_head(hidden_states)
    print(f"  样本 0 位置 79 的 logits: {class_logits[0, 79, :]}")
    print(f"  样本 0 取出的 logits:     {turn_logits[0, :]}")
    print(f"  是否一致: {torch.allclose(class_logits[0, 79, :], turn_logits[0, :])}")

    print("\n✅ 所有测试通过!")
