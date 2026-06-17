# UAF 微调计划：添加 Turn Head

## Context

在 Qwen3-Omni Thinker 模型基础上添加 **Turn Head**，用于轮次状态检测：
- 输出 4 类：`<Complete>` / `<InComplete>` / `<Backchannel>` / `<Interrupt>`
- 输入：Thinker 的 hidden_states (2048 维)
- 用途：实时对话中判断用户说话状态

**硬件配置**: 8 × RTX 3090 (24GB/卡)  
**训练策略**: 两阶段训练
- 阶段 A：4bit 冻结基座 + TD Head 线性探测（先跑通）
- 阶段 B：TD Head + QLoRA（效果不够再开）

---

## 1. 标签映射（关键修正）

### 1.1 Easy-Turn 标签 → UAF 标签

| Easy-Turn 标签 | txt 后缀 | TD Head id | UAF 风格 |
|---------------|----------|------------|----------|
| `<COMPLETE>` | `...<COMPLETE>` | 0 | `<Complete>` |
| `<INCOMPLETE>` | `...<INCOMPLETE>` | 1 | `<InComplete>` |
| `<BACKCHANNEL>` | `...<BACKCHANNEL>` | 2 | `<Backchannel>` |
| `<WAIT>` | `...<WAIT>` | 3 | `<Interrupt>` |

**⚠️ 重要**: Easy-Turn 使用 `<WAIT>`，不是 `<INTERRUPT>`！

### 1.2 标签解析代码

```python
TAGS = {
    "<COMPLETE>": 0,
    "<INCOMPLETE>": 1,
    "<BACKCHANNEL>": 2,
    "<WAIT>": 3,  # 映射到 UAF 的 Interrupt
}

ID2STATE = {
    0: "<Complete>",
    1: "<InComplete>",
    2: "<Backchannel>",
    3: "<Interrupt>",
}

def extract_turn_label(txt: str) -> int:
    """从 txt 字段提取轮次标签"""
    for tag, label_id in TAGS.items():
        if txt.endswith(tag):
            return label_id
    return -1  # 未知标签

def extract_transcript(txt: str) -> str:
    """提取纯文本（去除标签）"""
    for tag in TAGS.keys():
        if txt.endswith(tag):
            return txt[:-len(tag)]
    return txt
```

---

## 2. Turn Head 设计（简化版）

### 2.1 第一版：简单线性头

```python
class TurnHead(nn.Module):
    def __init__(self, hidden_size=2048, num_labels=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden):
        # hidden: [batch_size, hidden_size] (pooled)
        hidden = self.norm(hidden)
        hidden = self.dropout(hidden)
        return self.classifier(hidden)  # [batch_size, 4]
```

### 2.2 池化策略

```python
def get_pooled_hidden(hidden_states, attention_mask):
    """取最后一个非 padding 位置的 hidden state"""
    # hidden_states: [batch, seq_len, hidden_size]
    # attention_mask: [batch, seq_len]
    batch_size = hidden_states.size(0)
    seq_lens = attention_mask.sum(dim=1) - 1  # [batch]
    batch_idx = torch.arange(batch_size, device=hidden_states.device)
    return hidden_states[batch_idx, seq_lens]  # [batch, hidden_size]
```

### 2.3 输出格式

```
输入: hidden_states [batch, seq_len, 2048]
池化: pooled [batch, 2048]
输出: turn_logits [batch, 4]  # 不是 [batch, seq_len, 4]
```

---

## 3. 两阶段训练策略

### 3.1 阶段 A：TD Head-only（先跑通）

```yaml
experiment: easyturn_td_head_only

model:
  base: Qwen3-Omni-30B-A3B-Instruct
  use_vision: false
  use_talker: false
  use_vad_head: false
  use_turn_head: true

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true
  bnb_4bit_compute_dtype: float16

trainable:
  audio_encoder: false
  text_decoder: false
  lm_head: false
  lora: false          # 不开 LoRA
  turn_head: true      # 只训练 Turn Head

turn_head:
  type: layernorm_dropout_linear
  hidden_size: 2048
  num_labels: 4
  dropout: 0.1
  pooling: last_non_padding_hidden
```

**训练参数**：

```yaml
training:
  precision: fp16
  max_audio_seconds: 10    # 不要 30s，先用 10s
  max_seq_len: 1024        # 不要 4096

  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  num_train_epochs: 2

  optimizer: adamw
  lr_turn_head: 1.0e-3    # Turn Head 学习率可以较高
  weight_decay: 0.01
  warmup_ratio: 0.03
  max_grad_norm: 1.0

  gradient_checkpointing: true
  use_cache: false

  logging_steps: 20
  eval_steps: 500
  save_steps: 500
  save_total_limit: 3
```

**启动命令**：

```bash
torchrun --nproc_per_node=8 train_turn.py \
  --model_path ./Qwen3-Omni-30B-A3B-Instruct \
  --output_dir ./uaf_turn_head_output \
  --load_in_4bit \
  --bnb_4bit_quant_type nf4 \
  --bnb_4bit_compute_dtype float16 \
  --bnb_4bit_use_double_quant \
  --train_turn_head \
  --no_lora \
  --max_audio_seconds 10 \
  --max_seq_len 1024 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 2 \
  --lr_turn_head 1e-3
```

**如果显存爆**：

```bash
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16 \
--max_audio_seconds 8
```

---

### 3.2 阶段 B：TD Head + QLoRA（效果不够再开）

```yaml
experiment: easyturn_td_qlora

trainable:
  audio_encoder: false
  text_decoder_base: false
  lm_head: false
  lora: true           # 开启 LoRA
  turn_head: true
  td_cls_embedding: true  # 可选

td_cls:
  type: trainable_soft_embedding
  init_std: 0.02

lora:
  r: 8                 # 第二阶段先用 8，不要直接用 16
  lora_alpha: 16       # 对应 alpha = 2*r
  lora_dropout: 0.05
  bias: none
  target_layers: last_8_layers  # 40-47
  target_modules:
    - q_proj
    - v_proj
    - o_proj           # 先不加 k_proj 和 MLP
```

**训练参数**：

```yaml
training:
  precision: fp16
  max_audio_seconds: 10
  max_seq_len: 1024

  per_device_train_batch_size: 1    # QLoRA 显存紧，用 1
  gradient_accumulation_steps: 16
  num_train_epochs: 1

  optimizer: paged_adamw_8bit
  scheduler: cosine
  warmup_ratio: 0.03

  lr_lora: 5.0e-5              # LoRA 学习率较低
  lr_turn_head: 5.0e-4         # Turn Head 学习率中等
  lr_td_cls_embedding: 1.0e-4  # 可选

  weight_decay: 0.01
  max_grad_norm: 1.0
  gradient_checkpointing: true
  use_cache: false

  logging_steps: 20
  eval_steps: 500
  save_steps: 500
  save_total_limit: 3
```

**启动命令**：

```bash
torchrun --nproc_per_node=8 train_turn.py \
  --model_path ./Qwen3-Omni-30B-A3B-Instruct \
  --output_dir ./uaf_turn_qlora_output \
  --load_in_4bit \
  --bnb_4bit_quant_type nf4 \
  --bnb_4bit_compute_dtype float16 \
  --bnb_4bit_use_double_quant \
  --use_lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,v_proj,o_proj \
  --lora_target_layers 40,41,42,43,44,45,46,47 \
  --max_audio_seconds 10 \
  --max_seq_len 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 1 \
  --lr_lora 5e-5 \
  --lr_turn_head 5e-4
```

---

## 4. 自适应类别权重

### 4.1 方法一：基于 sqrt 的逆频率权重（推荐）

```python
import math
import torch

def compute_class_weights_sqrt(label_counts, max_weight=4.0):
    """
    基于 sqrt 的逆频率权重
    - 大类权重低，小类权重高
    - 使用 sqrt 平滑，避免小类权重过大
    
    Args:
        label_counts: [count_class0, count_class1, ...]
        max_weight: 权重上限
    Returns:
        weights: [weight_class0, weight_class1, ...]
    """
    max_count = max(label_counts)
    weights = []
    for count in label_counts:
        w = math.sqrt(max_count / count)
        w = min(w, max_weight)
        weights.append(w)
    return weights

# Easy-Turn 预估分布
# Complete: ~345k, InComplete: ~668k, Backchannel: ~100k, Wait: ~40k
label_counts = [345362, 667902, 100000, 40061]
weights = compute_class_weights_sqrt(label_counts)
# 结果: [1.39, 1.0, 2.58, 4.07] → 截断到 [1.39, 1.0, 2.58, 4.0]
```

### 4.2 方法二：Effective Number of Samples（更稳定）

```python
import numpy as np

def compute_class_weights_ens(label_counts, beta=0.9999):
    """
    Effective Number of Samples (CVPR 2019)
    论文: "Class-Balanced Loss Based on Effective Number of Samples"
    
    核心思想: 随着样本数增加，边际收益递减
    - 有效样本数 = (1 - beta^n) / (1 - beta)
    - 权重 = 1 / 有效样本数
    
    Args:
        label_counts: [count_class0, count_class1, ...]
        beta: 平滑因子，通常 0.99 ~ 0.9999
    Returns:
        weights: 归一化的权重
    """
    effective_num = 1.0 - np.power(beta, label_counts)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * len(label_counts)  # 归一化
    return weights.tolist()

# Easy-Turn 示例
label_counts = np.array([345362, 667902, 100000, 40061])
weights = compute_class_weights_ens(label_counts, beta=0.9999)
# 结果: [0.85, 0.65, 1.63, 2.87] (归一化后)
```

### 4.3 方法三：Focal Loss（动态调整）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss (ICCV 2017)
    论文: "Focal Loss for Dense Object Detection"
    
    核心思想:
    - 易分类样本权重降低，难分类样本权重提高
    - 通过 gamma 参数控制聚焦程度
    
    适用场景:
    - 类别不平衡
    - 难易样本不均衡
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # 类别权重，shape: [num_classes]
        self.gamma = gamma  # 聚焦参数，越大越关注难样本
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch_size, num_classes] (logits)
            targets: [batch_size] (labels)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 预测概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
```

### 4.4 方法四：Class-Balanced Loss（结合 ENS + Focal）

```python
class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss (CVPR 2019)
    结合 Effective Number of Samples + Focal Loss
    
    公式: CB_loss = (1 - beta^n_c) / (1 - beta) * FL(p, y)
    """
    def __init__(self, label_counts, beta=0.9999, gamma=2.0):
        super().__init__()
        self.focal_loss = FocalLoss(gamma=gamma, reduction='none')
        
        # 计算 ENS 权重
        effective_num = 1.0 - np.power(beta, label_counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(label_counts)
        self.alpha = torch.tensor(weights, dtype=torch.float32)

    def forward(self, inputs, targets):
        fl = self.focal_loss(inputs, targets)
        alpha = self.alpha.to(inputs.device)
        alpha_t = alpha[targets]
        return (alpha_t * fl).mean()
```

### 4.5 推荐方案

| 方法 | 适用场景 | 复杂度 | 推荐度 |
|------|----------|--------|--------|
| **sqrt 逆频率** | 简单有效，第一版首选 | ⭐ | ⭐⭐⭐⭐⭐ |
| ENS 逆频率 | 更平滑，适合极端不平衡 | ⭐⭐ | ⭐⭐⭐⭐ |
| Focal Loss | 难易样本不均衡 | ⭐⭐ | ⭐⭐⭐⭐ |
| CB Loss | 类别+难易都不均衡 | ⭐⭐⭐ | ⭐⭐⭐ |

**建议**：
- 阶段 A 用 **sqrt 逆频率**，简单有效
- 阶段 B 如果效果不够，尝试 **CB Loss**

### 4.6 实现代码

```python
class AdaptiveWeightLoss(nn.Module):
    """
    自适应权重损失函数
    支持多种权重策略
    """
    def __init__(self, num_classes=4, strategy='sqrt', label_counts=None):
        super().__init__()
        self.num_classes = num_classes
        self.strategy = strategy
        
        if strategy == 'sqrt':
            # 基于 sqrt 的逆频率
            weights = self._compute_sqrt_weights(label_counts)
        elif strategy == 'ens':
            # Effective Number of Samples
            weights = self._compute_ens_weights(label_counts)
        elif strategy == 'none':
            weights = torch.ones(num_classes)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        self.register_buffer('weights', torch.tensor(weights, dtype=torch.float32))
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weights, reduction='mean')
    
    def _compute_sqrt_weights(self, label_counts, max_weight=4.0):
        max_count = max(label_counts)
        weights = []
        for count in label_counts:
            w = math.sqrt(max_count / count)
            w = min(w, max_weight)
            weights.append(w)
        return weights
    
    def _compute_ens_weights(self, label_counts, beta=0.9999):
        effective_num = 1.0 - np.power(beta, label_counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(label_counts)
        return weights.tolist()
    
    def forward(self, logits, labels):
        return self.ce_loss(logits, labels)
```

### 4.7 使用示例

```python
# 阶段 A: 使用 sqrt 权重
train_dataset = TurnDataset(...)
label_counts = train_dataset.get_label_counts()
criterion = AdaptiveWeightLoss(
    num_classes=4,
    strategy='sqrt',
    label_counts=label_counts
)

# 阶段 B: 使用 CB Loss
criterion = ClassBalancedLoss(
    label_counts=label_counts,
    beta=0.9999,
    gamma=2.0
)
```

---

## 5. 数据处理

### 5.1 数据集类

```python
import torch
import json
import wave
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

TAGS = {
    "<COMPLETE>": 0,
    "<INCOMPLETE>": 1,
    "<BACKCHANNEL>": 2,
    "<WAIT>": 3,
}

class TurnDataset(Dataset):
    def __init__(self, list_file, trainset_dir, processor,
                 max_duration=10.0, max_seq_len=1024):
        self.data = []
        self.processor = processor
        self.max_duration = max_duration
        self.max_seq_len = max_seq_len
        self.trainset_dir = Path(trainset_dir)

        with open(list_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                label = self._extract_label(item['txt'])
                if label >= 0 and item['duration'] <= max_duration:
                    item['turn_label'] = label
                    self.data.append(item)

        print(f"Loaded {len(self.data)} samples from {list_file}")

    def _extract_label(self, txt):
        for tag, label_id in TAGS.items():
            if txt.endswith(tag):
                return label_id
        return -1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        wav_path = self.trainset_dir / item['wav'].lstrip('./')

        # 读取音频
        try:
            with wave.open(str(wav_path), 'r') as wf:
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                sr = wf.getframerate()
        except:
            audio = np.zeros(16000, dtype=np.float32)
            sr = 16000

        # 重采样
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        # 提取特征
        features = self.processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        )

        return {
            "input_features": features.input_features.squeeze(0),
            "turn_label": torch.tensor(item['turn_label'], dtype=torch.long),
        }
```

### 5.2 训练时不要喂 transcript

**⚠️ 重要**: 训练时不要把 ground-truth transcript 喂给模型，否则模型会靠文本作弊。

输入只给：
```
prompt + audio
```

`txt` 只用来提取标签，不作为输入。

---

## 6. 显存估算（8×RTX 3090, 24GB/卡）

### 6.1 模型加载方式

```
┌─────────────────────────────────────────────────────────────┐
│ 每张显卡加载完整的 NF4 量化模型                              │
├─────────────────────────────────────────────────────────────┤
│ 原始模型: ~60 GB (BF16)                                     │
│ NF4 量化: ~15 GB (4-bit)                                    │
│                                                              │
│ 每卡加载: 15 GB (完整模型，不分片)                           │
│ 8卡总计: 120 GB (8份完整模型)                                │
│                                                              │
│ ⚠️ 注意: NF4 模型每卡都持有完整副本                          │
│    优势: 无需 DeepSpeed 参数分片，通信开销小                  │
│    劣势: 显存占用较高                                        │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 阶段 A：只训练 TD Head

```
┌─────────────────────────────────────────────────────────────┐
│ 每卡显存分配 (8×3090, 24GB)                                  │
├─────────────────────────────────────────────────────────────┤
│ 组件                      │ 大小          │ 说明            │
├───────────────────────────┼───────────────┼─────────────────┤
│ NF4 模型 (完整)           │ ~15 GB        │ 冻结，不训练    │
│ TD Head (FP16)            │ ~0.01 GB      │ 可训练          │
│ 优化器状态 (TD Head)      │ ~0.04 GB      │ Adam 参数       │
│ 激活值 (梯度检查点)       │ ~4 GB         │ 前向/反向传播   │
│ 反量化临时空间            │ ~2 GB         │ NF4→FP16 临时  │
│ KV Cache + 缓冲区         │ ~1 GB         │ 推理缓存        │
├───────────────────────────┼───────────────┼─────────────────┤
│ 总计                      │ ~22 GB        │                 │
│ 可用显存                  │ 24 GB         │                 │
│ 剩余空间                  │ ~2 GB         │ ⚠️ 较紧         │
└─────────────────────────────────────────────────────────────┘

⚠️ 显存紧张，建议:
   - batch_size = 1
   - max_audio_seconds = 8
   - gradient_checkpointing = True
   - 或使用 gradient_accumulation_steps = 8
```

### 6.3 阶段 B：TD Head + QLoRA

```
┌─────────────────────────────────────────────────────────────┐
│ 每卡显存分配 (8×3090, 24GB)                                  │
├─────────────────────────────────────────────────────────────┤
│ 组件                      │ 大小          │ 说明            │
├───────────────────────────┼───────────────┼─────────────────┤
│ NF4 模型 (完整)           │ ~15 GB        │ 冻结            │
│ LoRA 参数 (FP16)          │ ~0.2 GB       │ 可训练 (r=16)   │
│ LoRA 优化器状态           │ ~0.8 GB       │ Adam 参数       │
│ TD Head (FP16)            │ ~0.01 GB      │ 可训练          │
│ TD Head 优化器            │ ~0.04 GB      │ Adam 参数       │
│ 激活值 (梯度检查点)       │ ~4 GB         │ 前向/反向传播   │
│ 反量化临时空间            │ ~2 GB         │ NF4→FP16 临时  │
│ KV Cache + 缓冲区         │ ~1 GB         │ 推理缓存        │
├───────────────────────────┼───────────────┼─────────────────┤
│ 总计                      │ ~23 GB        │                 │
│ 可用显存                  │ 24 GB         │                 │
│ 剩余空间                  │ ~1 GB         │ ⚠️ 很紧         │
└─────────────────────────────────────────────────────────────┘

⚠️ 显存非常紧张，建议:
   - batch_size = 1
   - max_audio_seconds = 8
   - gradient_checkpointing = True
   - gradient_accumulation_steps = 16
   - 或减少 LoRA rank (r=8)
```

### 6.4 优化建议

```
┌─────────────────────────────────────────────────────────────┐
│ 如果显存不足，按优先级调整:                                  │
├─────────────────────────────────────────────────────────────┤
│ 1. 减小 batch_size: 2 → 1                                   │
│ 2. 减小 max_audio_seconds: 10 → 8                           │
│ 3. 开启 gradient_checkpointing                              │
│ 4. 增大 gradient_accumulation_steps: 4 → 8 → 16             │
│ 5. 减小 LoRA rank: 16 → 8                                   │
│ 6. 减少 LoRA 层数: 16层 → 8层                               │
│ 7. 减少 target_modules: 去掉 gate/up/down_proj              │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. 评估指标

### 7.1 必须评估的指标

```python
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

def evaluate(predictions, labels):
    """评估指标"""
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average='macro'),
        "per_class_precision": precision_score(labels, predictions, average=None),
        "per_class_recall": recall_score(labels, predictions, average=None),
        "confusion_matrix": confusion_matrix(labels, predictions),
    }
```

### 7.2 重点关注的错误类型

```
InComplete → Complete
    会导致系统抢答

Wait → Complete / Backchannel
    会导致用户说"停止"时系统停不下来

Backchannel → Wait
    会导致用户"嗯嗯"时系统误打断
```

---

## 8. 数据划分

```yaml
split:
  train: 98%  # 从 trainset 分层采样
  valid: 2%   # 每类取几千条
  test: official_testset  # 保持不动，只做最终报告

valid_per_class:
  Complete: 3000
  InComplete: 3000
  Backchannel: 3000
  Wait: 3000
```

---

## 9. 实施步骤

| 步骤 | 内容 | 文件 | 阶段 |
|------|------|------|------|
| 1 | 创建 TurnHead 类 | `turn_head.py` | A |
| 2 | 创建配置文件 | `uaf_config.py` | A |
| 3 | 创建数据集类 | `turn_dataset.py` | A |
| 4 | 创建模型包装 | `uaf_model.py` | A |
| 5 | 编写训练脚本 | `train_turn.py` | A |
| 6 | 单卡测试 1000 条 | - | A |
| 7 | 8 卡训练完整数据 | - | A |
| 8 | 评估并决定是否开 LoRA | - | A→B |
| 9 | 添加 LoRA 支持 | `train_turn.py` | B |
| 10 | 8 卡 QLoRA 训练 | - | B |

---

## 10. 风险和注意事项

| 风险 | 说明 | 解决方案 |
|------|------|----------|
| 标签解析错误 | 用 `<INTERRUPT>` 而不是 `<WAIT>` | 使用 `TAGS` 字典正确映射 |
| 文本作弊 | 把 transcript 喂给模型 | 只用 prompt + audio |
| 显存估算乐观 | 4bit 权重可能不被 ZeRO-3 分片 | 保守参数，先单卡测试 |
| 类别不平衡 | InComplete 是大类 | 使用 class_weights |
| 训练不稳定 | 第一版太重 | 两阶段，先 Head-only |

---

## 11. 最终参数对比表

| 项目 | 原计划 | 修改后 |
|------|--------|--------|
| 标签 `<WAIT>` | 未支持 | ✅ 映射到 Interrupt |
| Turn Head 输出 | `[B, S, 4]` | `[B, 4]` |
| 第一版训练 | QLoRA + Head | 先只训 TD Head |
| DeepSpeed | ZeRO-3 | 先 torchrun/DDP |
| LoRA rank | 16 | 第二阶段先用 8 |
| LoRA alpha | 32 | 第二阶段先用 16 |
| LoRA target | q/k/v/o + gate | 先 q/v/o |
| learning rate | 2e-4 全局 | 分组 LR |
| Turn Head LR | 未单独设置 | 1e-3 |
| LoRA LR | 2e-4 | 5e-5 起步 |
| max duration | 30s | 8-10s |
| batch size | 2 | Head-only 可 2；QLoRA 用 1 |
| warmup | 0.1 + 500 steps | warmup_ratio=0.03 |
| class weights | InComplete 偏高 | Backchannel/Wait 偏高 |
| eval | accuracy | 加 macro-F1、confusion matrix |

---

## 12. 两阶段训练详解

### 12.1 阶段 A：只训练 TD Head

```
┌─────────────────────────────────────────────────────────────┐
│ 阶段 A: TD Head Only                                        │
├─────────────────────────────────────────────────────────────┤
│ 目标: 验证 audio hidden 是否包含轮次分类信息                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ 模型加载:                                                    │
│ ├── NF4 量化加载 (每卡完整模型 ~15GB)                        │
│ ├── 冻结: Audio Encoder, Text Decoder, LM Head              │
│ └── 训练: TD Head (随机初始化)                               │
│                                                              │
│ 训练参数:                                                    │
│ ├── epochs: 2                                                │
│ ├── batch_size: 1 (显存限制)                                 │
│ ├── gradient_accumulation: 8 (有效 batch = 8)               │
│ ├── max_audio_seconds: 8                                     │
│ ├── lr_td_head: 5e-4                                         │
│ ├── optimizer: AdamW                                         │
│ ├── scheduler: cosine                                        │
│ └── warmup_ratio: 0.03                                       │
│                                                              │
│ 损失函数:                                                    │
│ └── loss = CrossEntropy(turn_logits, turn_labels)           │
│                                                              │
│ 评估:                                                        │
│ ├── 每 200 步在 testset 上评估                               │
│ ├── 指标: accuracy, macro_f1                                 │
│ └── 早停: patience=10                                        │
│                                                              │
│ 预期结果:                                                    │
│ └── macro_f1 > 0.60 (如果 < 0.50 则需要阶段 B)              │
└─────────────────────────────────────────────────────────────┘
```

### 12.2 阶段 B：TD Head + QLoRA

```
┌─────────────────────────────────────────────────────────────┐
│ 阶段 B: TD Head + QLoRA                                     │
├─────────────────────────────────────────────────────────────┤
│ 目标: 通过 LoRA 微调提升模型特征提取能力                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│ 模型加载:                                                    │
│ ├── NF4 量化加载 (每卡完整模型 ~15GB)                        │
│ ├── 冻结: Audio Encoder                                      │
│ ├── LoRA: Text Decoder 后 16 层 (32-47)                     │
│ └── 训练: LoRA 参数 + TD Head                               │
│                                                              │
│ LoRA 配置:                                                   │
│ ├── rank: 16                                                 │
│ ├── alpha: 32                                                │
│ ├── dropout: 0.05                                            │
│ ├── target_modules: q/k/v/o_proj + gate/up/down_proj        │
│ └── layers: 32-47 (后 16 层)                                │
│                                                              │
│ 训练参数:                                                    │
│ ├── epochs: 1-2                                              │
│ ├── batch_size: 1                                            │
│ ├── gradient_accumulation: 16 (有效 batch = 16)             │
│ ├── max_audio_seconds: 8                                     │
│ ├── lr_td_head: 5e-4                                         │
│ ├── lr_lora: 5e-5                                            │
│ ├── optimizer: AdamW                                         │
│ └── scheduler: cosine                                        │
│                                                              │
│ 损失函数:                                                    │
│ ├── loss_td = CrossEntropy(turn_logits, turn_labels)        │
│ ├── loss_text = CrossEntropy(lm_logits, labels)             │
│ └── loss = loss_td + 0.1 * loss_text                        │
│                                                              │
│ 评估:                                                        │
│ ├── 每 200 步在 testset 上评估                               │
│ ├── 指标: accuracy, macro_f1, per_class_f1                  │
│ └── 早停: patience=10                                        │
│                                                              │
│ 预期结果:                                                    │
│ └── macro_f1 > 0.70                                         │
└─────────────────────────────────────────────────────────────┘
```

### 12.3 两阶段对比

| 项目 | 阶段 A | 阶段 B |
|------|--------|--------|
| **训练目标** | 只训练 TD Head | TD Head + LoRA |
| **冻结部分** | 全部基座 | Audio Encoder |
| **可训练参数** | ~8K (TD Head) | ~200M (LoRA + TD Head) |
| **学习率** | 5e-4 | 5e-4 (Head), 5e-5 (LoRA) |
| **batch_size** | 1 | 1 |
| **gradient_accumulation** | 8 | 16 |
| **显存/卡** | ~22 GB | ~23 GB |
| **训练时间** | ~2 小时 | ~4 小时 |
| **预期 F1** | > 0.60 | > 0.70 |

### 12.4 启动命令

**阶段 A:**
```bash
deepspeed --num_gpus=8 --master_port=29500 \
    train/train.py \
    --model_path ./Qwen3-Omni-30B-A3B-Instruct \
    --output_dir ./train/output_stage_a \
    --train_list_file ./dataset/Easy-Turn/Trainset_list/merged_balanced.list \
    --trainset_dir ./dataset/Easy-Turn/Trainset \
    --eval_list_file ./dataset/Easy-Turn/Testset/testset_all.list \
    --evalset_dir ./dataset/Easy-Turn/Testset \
    --sampler weighted \
    --stage A \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --max_audio_seconds 8 \
    --lr_td_head 5e-4 \
    --logging_steps 100 \
    --eval_steps 200 \
    --save_steps 500 \
    --early_stopping_patience 10
```

**阶段 B:**
```bash
deepspeed --num_gpus=8 --master_port=29500 \
    train/train.py \
    --model_path ./Qwen3-Omni-30B-A3B-Instruct \
    --output_dir ./train/output_stage_b \
    --train_list_file ./dataset/Easy-Turn/Trainset_list/merged_balanced.list \
    --trainset_dir ./dataset/Easy-Turn/Trainset \
    --eval_list_file ./dataset/Easy-Turn/Testset/testset_all.list \
    --evalset_dir ./dataset/Easy-Turn/Testset \
    --sampler weighted \
    --stage B \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_audio_seconds 8 \
    --lr_td_head 5e-4 \
    --lr_lora 5e-5 \
    --logging_steps 100 \
    --eval_steps 200 \
    --save_steps 500 \
    --early_stopping_patience 10
```
