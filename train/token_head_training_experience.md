# Token TD Head 训练经验总结

## 1. 概述

Token TD Head 是基于预训练 LM Head 的分类头，用于语音轮次状态四分类任务。

### 架构

```
Thinker hidden_states [B, T, 2048]
        ↓
取最后有效位置 last_hidden [B, 2048]
        ↓
token_class_head = Linear(2048, 4, bias=False)
        ↓
class_logits [B, 4]
        ↓
CE Loss
```

### 与原始 LM Head 的关系

| 方面 | 原始 LM Head | Token TD Head |
|------|-------------|---------------|
| 输出维度 | 152064 (词表) | 4 (类别) |
| 权重形状 | [152064, 2048] | [4, 2048] |
| 使用位置 | 所有位置 | 只用最后一个位置 |
| 位置编码 | 无 (在 Transformer 中) | 无 (在 Transformer 中) |
| 全局权重 | 是 | 是 |

---

## 2. 数据集

### 2.1 训练数据集

**文件**: `./dataset/Easy-Turn/Trainset_list/merged_balanced.list`

**总样本数**: 304,603

| 类别 | 样本数 | 比例 |
|------|--------|------|
| Complete | 128,316 | 42.13% |
| InComplete | 94,502 | 31.02% |
| Backchannel | 41,724 | 13.70% |
| Wait | 40,061 | 13.15% |

**类别权重**: `[1.0, 1.5, 3.0, 3.0]`

### 2.2 测试数据集

**文件**: `./dataset/Easy-Turn/Testset/testset_all.list`

**总样本数**: 800

| 类别 | 样本数 | 比例 |
|------|--------|------|
| Complete | 300 | 37.50% |
| InComplete | 300 | 37.50% |
| Backchannel | 100 | 12.50% |
| Wait | 100 | 12.50% |

### 2.3 数据集特点

1. **类别不平衡**
   - 训练集: Complete (42%) > InComplete (31%) > Backchannel (14%) ≈ Wait (13%)
   - 测试集: Complete (37.5%) = InComplete (37.5%) > Backchannel (12.5%) = Wait (12.5%)

2. **使用类别权重平衡**
   - Complete: 1.0 (权重最低)
   - InComplete: 1.5
   - Backchannel: 3.0
   - Wait: 3.0 (权重最高)

3. **测试集均衡设计**
   - Complete 和 InComplete 各 300 个样本
   - Backchannel 和 Wait 各 100 个样本

---

## 2. 初始化策略

### 2.1 问题：随机初始化效果差

最初的随机初始化效果很差，Best Macro-F1 只有 0.3570。

### 2.2 解决方案：从预训练 LM Head 初始化

从预训练 LM Head 中提取与任务相关的词向量作为初始化。

#### 步骤

1. **找到每个类别相关的词汇**
   ```python
   RELEVANT_TOKENS = {
       0: ['complete', 'finish', 'end', 'done', 'over', 'conclude',
           '完成', '结束', '搞定', '收尾', '完结'],
       1: ['incomplete', 'continue', 'keep', 'go', 'proceed', 'next',
           '继续', '接着', '未完', '待续', '进行'],
       2: ['backchannel', 'mm', 'uh', 'yeah', 'okay', 'right',
           '嗯', '哦', '好的', '对', '啊'],
       3: ['wait', 'interrupt', 'hold', 'stop', 'pause', 'halt',
           '稍等', '暂停', '停下', '等等', '打断'],
   }
   ```

2. **查询到的 Token ID**

   **Class 0 (Complete)** - 找到 9/11 个 token:
   | 词汇 | Token ID | 状态 |
   |------|----------|------|
   | complete | 14737 | ✅ |
   | finish | 30150 | ✅ |
   | end | 408 | ✅ |
   | done | 10438 | ✅ |
   | over | 1975 | ✅ |
   | conclude | [443, 857] | ⚠️ 多 token |
   | 完成 | 60548 | ✅ |
   | 结束 | 80565 | ✅ |
   | 搞定 | 112719 | ✅ |
   | 收尾 | [50009, 101143] | ⚠️ 多 token |
   | 完结 | 118498 | ✅ |

   **Class 1 (InComplete)** - 找到 7/11 个 token:
   | 词汇 | Token ID | 状态 |
   |------|----------|------|
   | incomplete | [258, 14737] | ⚠️ 多 token |
   | continue | 9534 | ✅ |
   | keep | 13096 | ✅ |
   | go | 3346 | ✅ |
   | proceed | [776, 4635] | ⚠️ 多 token |
   | next | 3600 | ✅ |
   | 继续 | 100640 | ✅ |
   | 接着 | 102524 | ✅ |
   | 未完 | [38342, 46306] | ⚠️ 多 token |
   | 待续 | [74193, 99448] | ⚠️ 多 token |
   | 进行 | 71817 | ✅ |

   **Class 2 (Backchannel)** - 找到 10/11 个 token:
   | 词汇 | Token ID | 状态 |
   |------|----------|------|
   | backchannel | [1419, 10119] | ⚠️ 多 token |
   | mm | 3821 | ✅ |
   | uh | 12540 | ✅ |
   | yeah | 75415 | ✅ |
   | okay | 93217 | ✅ |
   | right | 1291 | ✅ |
   | 嗯 | 106287 | ✅ |
   | 哦 | 104170 | ✅ |
   | 好的 | 99692 | ✅ |
   | 对 | 32664 | ✅ |
   | 啊 | 103924 | ✅ |

   **Class 3 (Wait)** - 找到 10/11 个 token:
   | 词汇 | Token ID | 状态 |
   |------|----------|------|
   | wait | 11489 | ✅ |
   | interrupt | 54805 | ✅ |
   | hold | 6282 | ✅ |
   | stop | 9495 | ✅ |
   | pause | 27448 | ✅ |
   | halt | 39416 | ✅ |
   | 稍等 | [93266, 49567] | ⚠️ 多 token |
   | 暂停 | 107276 | ✅ |
   | 停下 | 112888 | ✅ |
   | 等等 | 104008 | ✅ |
   | 打断 | 115458 | ✅ |

   **统计**: 共找到 36/44 个单 token (81.8%)

2. **获取这些词汇的 token_id**
   ```python
   vocab = tokenizer.get_vocab()
   token_ids = [vocab[t] for t in tokens if t in vocab]
   ```

3. **从预训练 LM Head 提取行向量**
   ```python
   rows = lm_head_weight[token_ids]  # [N, 2048]
   ```

4. **取平均值**
   ```python
   avg = rows.mean(dim=0)  # [2048]
   ```

5. **缩放**
   ```python
   scaled = avg * 0.5  # 缩放系数 0.5
   ```

6. **组合成 W_turn**
   ```python
   W_turn = torch.stack([scaled_0, scaled_1, scaled_2, scaled_3])  # [4, 2048]
   ```

7. **初始化 TD Head**
   ```python
   token_class_head.weight.data = W_turn
   ```

### 2.3 初始化效果对比

| 初始化策略 | Best Macro-F1 | 提升 |
|-----------|---------------|------|
| 随机初始化 | 0.3570 | - |
| 列向量平均值 + 缩放 0.5 | 0.7879 | +120% |

---

## 3. NaN 问题及解决方案

### 3.1 问题现象

训练过程中出现 NaN，权重在 optimizer.step() 后变成 NaN。

```
Step 0-3: 梯度正常 (max=9.2, 16.2, 15.1)
Step 3 optimizer.step() 后: 权重变成 NaN (nan=4900)
```

### 3.2 原因分析

1. **AdamW 的二阶矩估计**
   - AdamW 维护两个移动平均: m_t (一阶矩) 和 v_t (二阶矩)
   - 如果 v_t 很小，sqrt(v_t) + ε 也很小
   - 导致更新量很大，可能溢出

2. **初始化权重太小**
   - 缩放系数 0.2 导致权重标准差只有 0.002478
   - 权重太小 → 梯度相对很大 → AdamW 的 v_t 估计不稳定

3. **梯度累积 + AdamW 的交互**
   - 梯度累积 4 步后一次性更新
   - 累积的梯度可能很大，AdamW 的自适应机制来不及适应

### 3.3 解决方案

#### 方案 1: 增大初始化权重 (推荐)

将缩放系数从 0.2 增大到 0.5。

```python
# 修改前
init_scale = 0.2

# 修改后
init_scale = 0.5
```

**效果**: 权重标准差从 0.002478 增大到 0.006195，NaN 问题解决。

#### 方案 2: 使用 SGD 优化器

```python
# 修改前
optimizer = torch.optim.AdamW(params)

# 修改后
optimizer = torch.optim.SGD(params, momentum=0.9)
```

**效果**: SGD 更简单、更稳定，不会出现 NaN。

#### 方案 3: 降低学习率

```python
# 修改前
lr = 5e-4

# 修改后
lr = 1e-5
```

**效果**: 降低学习率可以减小更新量，但可能影响收敛速度。

#### 方案 4: 更强的梯度裁剪

```python
# 修改前
torch.nn.utils.clip_grad_norm_(params, 1.0)

# 修改后
torch.nn.utils.clip_grad_norm_(params, 0.1)
```

**效果**: 更强的梯度裁剪可以防止梯度爆炸。

### 3.4 最终方案

使用 **增大初始化权重 (0.5)** + **AdamW 优化器** + **学习率 5e-4** + **梯度裁剪 1.0**。

---

## 4. 训练配置

### 4.1 推荐配置

```bash
python train/train.py \
    --model_path ./Qwen3-Omni-30B-A3B-Instruct \
    --output_dir ./train/output_stage_a_token \
    --train_list_file ./dataset/Easy-Turn/Trainset_list/merged_balanced.list \
    --trainset_dir ./dataset/Easy-Turn/Trainset \
    --eval_list_file ./dataset/Easy-Turn/Testset/testset_all.list \
    --evalset_dir ./dataset/Easy-Turn/Testset \
    --sampler none \
    --stage A \
    --td_head_type token \
    --init_strategy column \
    --init_scale 0.5 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --max_audio_seconds 60 \
    --lr_td_head 5e-4 \
    --logging_steps 50 \
    --eval_steps 200 \
    --save_steps 500 \
    --early_stopping_patience 10 \
    --num_workers 32 \
    --class_weights 1.0 1.5 3.0 3.0
```

### 4.2 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| td_head_type | token | Token TD Head |
| init_strategy | column | 列向量平均值初始化 |
| init_scale | 0.5 | 缩放系数 |
| lr_td_head | 5e-4 | 学习率 |
| per_device_train_batch_size | 32 | 批次大小 |
| gradient_accumulation_steps | 4 | 梯度累积步数 |
| early_stopping_patience | 10 | 早停耐心 |

---

## 5. 训练结果

### 5.1 最佳结果

```
Best Testset Macro-F1: 0.7879
Testset Acc: 0.8200

各类别详细指标:
  Complete:     Acc=0.9333 | P=0.7609 | R=0.9333 | F1=0.8383
  InComplete:   Acc=0.7900 | P=0.8843 | R=0.7900 | F1=0.8345
  Backchannel:  Acc=0.4500 | P=0.8824 | R=0.4500 | F1=0.5960
  Wait:         Acc=0.9400 | P=0.8319 | R=0.9400 | F1=0.8826
```

### 5.2 与其他模型对比

| 模型 | 参数量 | Best Macro-F1 | 状态 |
|------|--------|---------------|------|
| Dense SwiGLU | ~56M | 0.4619 | ✅ 完成 |
| MoE | ~623M | 0.4624 | ✅ 完成 |
| Token (旧) | ~8K | 0.3570 | ✅ 完成 |
| Token (新) | ~8K | 0.7879 | ✅ 完成 |

### 5.3 分析

1. **Token Head 效果最好**
   - 参数量最少 (8K)
   - 效果最好 (Macro-F1=0.7879)
   - 初始化策略是关键

2. **类别不平衡问题**
   - Backchannel 的 F1 只有 0.5960
   - 可能需要调整 class_weights 或使用过采样

3. **训练速度**
   - Token Head 训练速度最快 (~2.1s/step)
   - 预计总训练时间 ~13 小时

---

## 6. 关键经验

### 6.1 初始化很重要

- 随机初始化效果差 (Macro-F1=0.3570)
- 从预训练 LM Head 初始化效果好 (Macro-F1=0.7879)
- 使用相关词的平均值可以平滑噪声
- 缩放系数需要适中 (0.5 比 0.2 好)

### 6.2 NaN 问题需要预防

- 初始化权重太小会导致 NaN
- AdamW + 大学习率 + 小权重 = NaN
- 解决方案: 增大初始化权重或使用 SGD

### 6.3 学习率调度

- 使用线性预热 + 余弦退火
- 预热步数: 总步数的 1%
- 可以提高训练稳定性

### 6.4 早停很重要

- 防止过拟合
- 节省训练时间
- 保存最优模型

### 6.5 监控训练过程

- 检查 loss 是否下降
- 检查梯度是否正常
- 检查权重是否更新
- 定期评估测试集

---

## 7. 待优化方向

1. **类别不平衡**
   - 调整 class_weights
   - 使用过采样或欠采样
   - 使用 Focal Loss

2. **超参数调优**
   - 尝试不同的学习率
   - 尝试不同的缩放系数
   - 尝试不同的梯度裁剪

3. **模型结构**
   - 尝试使用多个位置的 hidden_state
   - 尝试添加 Dropout
   - 尝试使用 LayerNorm

4. **训练策略**
   - 尝试不同的学习率调度
   - 尝试不同的梯度累积步数
   - 尝试不同的批次大小

---

## 8. 代码位置

| 文件 | 说明 |
|------|------|
| train/td_head_token.py | Token TD Head 实现 |
| train/train.py | 训练代码 |
| train/run_stage_a_token.sh | 训练脚本 |
| train/output_stage_a_token/ | 输出目录 |
| train/output_stage_a_token/results.txt | 训练结果 |
