# Qwen-Style TD Head 分类头结构

## 概述

两种分类头都基于 Qwen3-Omni-30B-A3B 的 Attention + FFN 架构，使用 **Learned Label Token** 作为分类锚点。

**共同特点**:
- GQA (Grouped Query Attention) 与 Qwen3-Omni 一致
- RoPE 位置编码
- QK Norm (RMSNorm)
- 支持 KV Cache 流式推理
- Learned Label Token 拼接到序列末尾进行分类

---

## 1. MoE 版本 (默认)

**配置**: `--ffn_type moe`
**参数量**: ~623M

### 整体架构

```
输入: audio_hidden [B, A, 2048]
         │
    [拼接 learned label token]
    x = [audio_hidden; label_token]  →  [B, A+1, 2048]
         │
    ╔═══════════════════════════════════════════════════════╗
    ║  Decoder Layer × N (默认 N=1)                          ║
    ║                                                       ║
    ║    x ──→ RMSNorm ──→ GQA ──→ + x (residual)          ║
    ║                              │                        ║
    ║    ──→ RMSNorm ──→ MoE FFN ──→ + (residual)          ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
         │
    取最后一个 token (label token 位置)
         │
    RMSNorm → Linear(2048 → 4) → turn_logits [B, 4]
```

### GQA 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| hidden_size | 2048 | 隐藏维度 |
| num_heads | 32 | Query 头数 |
| num_kv_heads | 4 | Key/Value 头数 |
| head_dim | 128 | 每头维度 |
| rope_theta | 1,000,000 | RoPE 频率 |

**投影层**:
```
q_proj: 2048 → 4096  (32 × 128)
k_proj: 2048 → 512   (4 × 128)
v_proj: 2048 → 512   (4 × 128)
o_proj: 4096 → 2048
```

**注意力计算**:
1. Q, K, V 投影后 reshape 为多头
2. Q, K 应用 RMSNorm (QK Norm)
3. 应用 RoPE 位置编码
4. K, V 扩展以匹配 Q 头数 (4 → 32, 每组共享 8 个 KV 头)
5. Causal mask + Padding mask
6. Softmax + Attention 输出

### MoE FFN 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| num_experts | 128 | 专家总数 |
| top_k | 8 | 每 token 选择的专家数 |
| intermediate_size | 768 | 每个专家的中间维度 |

**路由机制**:
```
router: 2048 → 128  (计算每个 token 对 128 个专家的偏好)
         │
    softmax → top-8 选择
         │
    权重重归一化 (sum=1)
         │
    分发到对应专家计算
```

**每个 Expert (SwiGLU FFN)**:
```
gate_proj: 2048 → 768
up_proj:   2048 → 768
down_proj: 768  → 2048

FFN(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
```

**Per-Expert Dispatch**:
- 避免 gather-all 方式导致 OOM
- 逐专家处理，找到路由到该专家的所有 token
- 计算后加权累加到输出

### 参数统计

| 组件 | 参数量 |
|------|--------|
| Label Token | 2,048 |
| GQA (q/k/v/o_proj + norm) | ~8.4M |
| MoE Router | 262K |
| MoE Experts (128 × 3 linear) | ~589M |
| LayerNorm (×2) | 4,096 |
| Final Norm | 2,048 |
| Classifier | 8,196 |
| **总计** | **~623M** |

---

## 2. Dense 版本

**配置**: `--ffn_type dense`
**参数量**: ~56M

### 整体架构

```
输入: audio_hidden [B, A, 2048]
         │
    [拼接 learned label token]
    x = [audio_hidden; label_token]  →  [B, A+1, 2048]
         │
    ╔═══════════════════════════════════════════════════════╗
    ║  Decoder Layer × N (默认 N=1)                          ║
    ║                                                       ║
    ║    x ──→ RMSNorm ──→ GQA ──→ + x (residual)          ║
    ║                              │                        ║
    ║    ──→ RMSNorm ──→ Dense SwiGLU ──→ + (residual)     ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
         │
    取最后一个 token (label token 位置)
         │
    RMSNorm → Linear(2048 → 4) → turn_logits [B, 4]
```

### GQA 参数

与 MoE 版本完全相同，见上文。

### Dense SwiGLU FFN 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| intermediate_size | 6144 | 中间维度 (3× hidden_size) |

**FFN 层**:
```
gate_proj: 2048 → 6144
up_proj:   2048 → 6144
down_proj: 6144 → 2048

FFN(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
```

**与 MoE 的区别**:
- 没有路由，所有 token 共享同一个 FFN
- 参数量从 ~589M (128 experts) 降到 ~38M (单个 FFN)
- 计算更简单，不需要 per-expert dispatch

### 参数统计

| 组件 | 参数量 |
|------|--------|
| Label Token | 2,048 |
| GQA (q/k/v/o_proj + norm) | ~8.4M |
| Dense SwiGLU FFN | ~37.7M |
| LayerNorm (×2) | 4,096 |
| Final Norm | 2,048 |
| Classifier | 8,196 |
| **总计** | **~56M** |

---

## 3. 对比总结

| 特性 | MoE 版本 | Dense 版本 |
|------|---------|-----------|
| **参数量** | ~623M | ~56M |
| **FFN 类型** | MoE (128 experts, top-8) | Dense SwiGLU |
| **FFN 中间维度** | 768 (per expert) | 6144 |
| **路由开销** | 有 (router + dispatch) | 无 |
| **训练速度** | ~2.3s/step | ~1.8s/step |
| **显存占用** | 较高 | 较低 |
| **表达能力** | 理论更强 (稀疏激活) | 一般 |

### FFN 计算对比

**MoE**:
```
每个 token 只激活 8/128 = 6.25% 的专家参数
实际计算量: 8 × (768 × 3) = 18,432 ops/token
```

**Dense**:
```
每个 token 激活全部 FFN 参数
实际计算量: 6144 × 3 = 18,432 ops/token
```

> 注: 两者的每 token 计算量相近，但 MoE 的参数量更大，理论上可以学到更多专业化知识。

---

## 4. 训练配置

| 配置 | MoE | Dense |
|------|-----|-------|
| **脚本** | `run_stage_a_qwen.sh` | `run_stage_a_qwen_dense.sh` |
| **ffn_type** | moe (默认) | dense |
| **Batch Size** | 32 | 32 |
| **Grad Accum** | 4 | 4 |
| **学习率** | 5e-4 | 5e-4 |
| **Epochs** | 2 | 2 |
| **Class Weights** | [1.0, 1.5, 3.0, 3.0] | [1.0, 1.5, 3.0, 3.0] |

---

## 5. 当前实验结果

| 模型 | 参数量 | Best Macro-F1 | 状态 |
|------|--------|--------------|------|
| Dense SwiGLU | ~56M | **0.4619** | ✅ 完成 |
| MoE | ~623M | 待定 | 🔄 训练中 (F1=0.4299 @ step 400) |
