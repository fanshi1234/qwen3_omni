# UAF Turn Head Training

## 方案说明

**双头输出架构：**

```
输入：prompt + audio（不输入真实转写）
     ↓
[Qwen3-Omni Base Model]
     ↓
┌────────────────────────────────────┐
│  LM Head (ASR)  │  TD Head (分类)  │
│  预测转写文本    │  四分类 logits   │
└────────────────────────────────────┘

真实转写只作为 ASR target，模型看到的只有音频和任务说明。
```

**四分类标签：**
- `<COMPLETE>`: 话轮完成
- `<INCOMPLETE>`: 话轮未完成
- `<BACKCHANNEL>`: 反馈信号
- `<WAIT>`: 等待/打断

**损失函数：**
```
L_total = L_td + λ * L_asr
```
- `L_td`: TD Head 的交叉熵损失（带类别权重）
- `L_asr`: LM Head 的语言模型损失
- `λ`: ASR loss 权重（默认 0.1）

## 文件结构

```
train/
├── config.py       # 配置定义
├── dataset.py      # 数据集加载
├── loss.py         # 损失函数
├── turn_head.py    # TD Head 模型
├── train.py        # 训练脚本
├── run.sh          # 启动脚本
└── README.md       # 说明文档
```

## 快速开始

### 阶段 A: Head-only 线性探测

```bash
cd /data1/wgy/qwen
bash train/run.sh
```

### 阶段 B: Head + LoRA

```bash
cd /data1/wgy/qwen
bash train/run.sh --stage B --use_lora
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | Qwen3-Omni-30B-A3B-Instruct | 基座模型路径 |
| `--output_dir` | ./train/output | 输出目录 |
| `--stage` | A | 训练阶段 (A/B) |
| `--num_train_epochs` | 2 | 训练轮数 |
| `--per_device_train_batch_size` | 2 | 批次大小 |
| `--gradient_accumulation_steps` | 8 | 梯度累积步数 |
| `--lr_turn_head` | 1e-3 | TD Head 学习率 |
| `--lambda_asr` | 0.1 | ASR loss 权重 |
| `--max_audio_seconds` | 10.0 | 最大音频时长 |
| `--use_lora` | false | 是否使用 LoRA |
| `--lora_rank` | 8 | LoRA rank |
| `--lora_alpha` | 16 | LoRA alpha |
| `--lr_lora` | 5e-5 | LoRA 学习率 |

## 数据格式

训练集 list 文件格式 (JSONL):
```json
{
    "task": "<TRANSCRIBE> <BACKCHANNEL> <COMPLETE>",
    "key": "sample_001",
    "wav": "./path/to/audio.wav",
    "txt": "文本内容<COMPLETE>",
    "lang": "<CN>",
    "duration": 2.04,
    "state": "0"
}
```

## 类别权重

使用 sqrt 逆频率权重处理类别不平衡：

| 类别 | 样本数 | 权重 |
|------|--------|------|
| Complete | 423,678 | 1.0 |
| InComplete | 712,404 | 1.0 |
| Backchannel | 41,724 | 3.18 |
| Wait | 40,061 | 3.25 |

## 评估指标

- Accuracy
- Macro-F1
- Per-class Precision/Recall
- Confusion Matrix

## 输出格式

训练输出保存在 `output/` 目录：

```
output/
├── config.json           # 训练配置
├── train.log             # 训练日志
├── checkpoint-500/       # 中间检查点
│   └── turn_head.pt
├── checkpoint-1000/
│   └── turn_head.pt
└── final/
    └── turn_head.pt      # 最终模型
```

## 注意事项

1. **不要喂 transcript**: 训练时只用 `prompt + audio`，不用 `txt` 字段作为输入
2. **ASR target**: 真实转写只作为 LM Head 的训练目标
3. **标签来源**: 从 `txt` 后缀提取标签 (`<COMPLETE>`, `<INCOMPLETE>`, `<BACKCHANNEL>`, `<WAIT>`)
4. **显存管理**: 如果显存不足，减小 `batch_size` 或 `max_audio_seconds`
5. **两阶段训练**: 先跑阶段 A，效果不够再开阶段 B
