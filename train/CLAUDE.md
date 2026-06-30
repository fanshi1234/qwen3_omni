# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

UAF (Understanding Audio in Full-duplex) 双头训练项目，基于 Qwen3-Omni-30B-A3B-Instruct 多模态大模型，添加 TD Head 实现语音轮次状态四分类：

- `COMPLETE` (0) - 用户说完，系统可以回答
- `INCOMPLETE` (1) - 用户没说完，系统继续听
- `BACKCHANNEL` (2) - 用户附和/反馈
- `WAIT` (3) - 用户要求暂停/停止

## 训练阶段

| 阶段 | 说明 | LoRA | 目标 |
|------|------|------|------|
| Stage A | 只训练 TD Head | 否 | 冻结基座，训练分类头 |
| Stage B | TD Head + LoRA | 是 (rank=16, alpha=32) | 微调后 16 层 Attention |
| Stage C | TD Head + LoRA + Audio Projector | 是 | 待实现 |

## 常用命令

```bash
# 环境
conda activate qwen

# Stage A 训练
bash train/run.sh A

# Stage B 训练 (需先完成 Stage A)
bash train/run.sh B

# 独立评估
python train/eval_best.py

# 合并数据集
python train/merge_datasets.py
python train/merge_testset.py
```

## 代码架构

```
train.py          - 主入口：UAFModel 类、训练循环、评估逻辑
config.py         - 全局常量：SYSTEM_PROMPT、LABEL2ID、音频 token ID
td_head.py        - TD Head 定义：Attention Pooling + 4 分类器
dataset.py        - 数据集加载：TurnDataset、采样策略
loss.py           - 损失函数库：FocalLoss、ClassBalancedLoss 等
eval_best.py      - 独立评估脚本
merge_datasets.py - 合并训练集
merge_testset.py  - 合并测试集
```

## 模型架构

```
输入: [system_prompt] [audio_tokens] [text_target]
         |
  Qwen3-Omni-30B-A3B (冻结, FP16, device_map=auto)
         |
    hidden_states [-1]
         |
    +------+------+------+
    |                    |
    LM Head (ASR)    TD Head (分类)
    (Stage B/C)        (所有 Stage)
```

- TD Head: Attention Pooling → LayerNorm → Dropout → Linear(H, 4)
- LoRA: 仅应用于后 16 层 (layer 32-47) 的 q/k/v/o_proj
- 损失: L = L_td + 0.1 * L_asr (Stage A 只有 L_td)

## 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--stage` | A | 训练阶段 |
| `--use_lora` | false | 启用 LoRA |
| `--lora_rank` | 8 | LoRA 秩 |
| `--lr_td_head` | 5e-4 | TD Head 学习率 |
| `--lr_lora` | 5e-5 | LoRA 学习率 |
| `--per_device_train_batch_size` | 1 | 每卡 batch size |
| `--gradient_accumulation_steps` | 16 | 梯度累积 |
| `--max_audio_seconds` | 10.0 | 最大音频时长 |
| `--sampler` | weighted | 采样策略 |
| `--load_td_head` | None | 加载 Stage A 权重 |

## 输出目录结构

```
output_stage_{a,b}/
├── best_checkpoint/      # 最优模型 (td_head.pt + adapter_model.safetensors)
├── checkpoint-{N}/       # 定期保存的 checkpoint
├── final/                # 最终模型
├── training_config.json  # 训练参数 (自动生成)
├── run_train.sh          # 启动脚本 (自动生成)
├── results.txt           # 结果摘要 (自动生成)
└── train.log             # 训练日志
```

## 数据格式

训练集 list 文件 (JSONL 格式):
```json
{"wav": "path/to/audio.wav", "txt": "用户文本 <COMPLETE>", "duration": 3.5, "key": "sample_001"}
```

标签从 `txt` 字段末尾提取，支持: `<COMPLETE>`, `<INCOMPLETE>`, `<BACKCHANNEL>`, `<WAIT>`

## 注意事项

- 基座模型使用 FP16 + device_map="auto" 分布到 8 张 3090 (24GB/卡)
- 当前未使用 DeepSpeed (NF4 量化兼容问题)
- Stage B 需要加载 Stage A 的 TD Head 权重 (`--load_td_head`)
- 音频在模型内部加载和重采样到 16kHz，不在 Dataset 中处理
