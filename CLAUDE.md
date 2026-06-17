# Qwen3-Omni UAF 项目规范

> 本文件是 `/data1/wgy/qwen` 的项目级规范，供 Claude Code 在本项目中读取和遵守。
> 如本文件与源码、`config.json`、训练脚本冲突，以实际代码和配置为准，并在回复中指出冲突。

---

## 1. 项目目标

本项目基于 **Qwen3-Omni-30B-A3B-Instruct** 构建 UAF（Unified Audio Framework）微调方案，在保留原模型多模态能力的基础上新增 **Turn Head**，用于语音轮次状态检测。

Turn Head 需要预测 4 类状态：

| ID | 标签            | 含义         |
| -: | ------------- | ---------- |
|  0 | `Complete`    | 用户已说完，可以回复 |
|  1 | `InComplete`  | 用户尚未说完，应等待 |
|  2 | `Interrupt`   | 用户被打断或存在插话 |
|  3 | `Backchannel` | 用户为反馈词/附和语 |

默认硬件与训练方案：

* 硬件：8 × RTX 3090，24GB/卡
* 量化：4-bit QLoRA，NF4
* 并行：DeepSpeed ZeRO-3
* 微调范围：默认只训练 Turn Head + Text Decoder 后 8 层 LoRA
* 默认不进行 BF16 全量微调

---

## 2. Claude 工作规则

Claude 在本项目中必须遵守：

1. **先读后改**：修改前先查看相关源码、配置和调用链。
2. **最小变更**：只修改当前任务需要的文件，不做无关重构。
3. **保护资产**：不得删除、移动或覆盖模型权重、原始数据集和官方仓库源码。
4. **显式验证**：完成代码修改后，说明运行过哪些测试；未运行则说明原因。
5. **不擅自提交**：除非用户明确要求，不执行 `git commit`、`git push`。
6. **不隐藏失败**：命令失败时报告失败命令、关键错误和下一步建议。
7. **不写入私密信息**：不得把 token、密码、私有链接、账号凭据写入代码或文档。
8. **沉淀长期经验**：发现长期有效的命令、坑点、架构约束时，建议写入 `CLAUDE.md`、`CLAUDE.local.md` 或 Auto memory。

---

## 3. 项目目录职责

默认工作目录：

```bash
cd /data1/wgy/qwen
```

| 路径                                | 职责           | 规则                             |
| --------------------------------- | ------------ | ------------------------------ |
| `CLAUDE.md`                       | 项目规范         | 当前文件，作为 Claude 项目级行为规范         |
| `.claude/`                        | Claude 配置    | 存放计划、规则、技能等辅助文件                |
| `.claude/plans/turn_head_plan.md` | 实施计划         | Turn Head 微调详细计划               |
| `Qwen3-Omni/`                     | 官方代码仓库       | 默认只读；不要无关修改                    |
| `Qwen3-Omni-30B-A3B-Instruct/`    | 模型权重         | 只读；不要删除、移动、覆盖                  |
| `dataset/Easy-Turn/`              | 数据集          | 原始数据只读；派生文件另存                  |
| `turn_head.py`                    | 模型代码         | Turn Head 定义                   |
| `uaf_config.py`                   | 配置代码         | 训练/推理配置                        |
| `turn_dataset.py`                 | 数据代码         | Easy-Turn 数据加载                 |
| `uaf_model.py`                    | 模型包装         | Qwen3-Omni + Turn Head wrapper |
| `train_turn.py`                   | 训练入口         | 单卡/多卡训练                        |
| `test_turn.py`                    | 推理评估         | 单音频推理和评估                       |
| `ds_config.json`                  | DeepSpeed 配置 | ZeRO-3 训练配置                    |
| `output/`                         | 训练产物         | 可生成；删除前必须确认                    |

---

## 4. 模型架构约束

基座模型：

* 类：`Qwen3OmniMoeForConditionalGeneration`
* hidden size：`2048`
* Audio 输入采样率：`16000`
* 语音输出采样率：`24000`
* 具体模型参数以 `Qwen3-Omni-30B-A3B-Instruct/config.json` 为准

UAF Turn Head 流程：

```text
输入音频 16kHz
  -> Audio Encoder        冻结
  -> Text Decoder         4-bit NF4；默认仅后 8 层 LoRA
  -> hidden_states        [batch, seq_len, 2048]
  -> Turn Head            新增可训练分类头
  -> turn_logits          [batch, seq_len, 4]
```

未经用户确认，不要：

* 改成全参数微调
* 改成 BF16 全量训练
* 修改 Talker 语音生成路径
* 修改官方模型权重目录
* 把 processor、dataset、trainer 逻辑混入模型类

---

## 5. 数据集规范

数据集根目录：

```text
dataset/Easy-Turn/
```

训练/测试目录：

| 路径                     | 说明                    |
| ---------------------- | --------------------- |
| `Trainset/`            | 训练集 tar 包             |
| `Testset/complete/`    | complete 测试样本         |
| `Testset/incomplete/`  | incomplete 测试样本       |
| `Testset/backchannel/` | backchannel 测试样本      |
| `Testset/wait/`        | wait/interrupt 相关测试样本 |

数据加载代码必须：

* 支持从 `txt` 后缀标签和 `state` 字段解析类别。
* 返回音频路径、文本、标签、样本 key 等必要字段。
* 对缺失文件、非法标签、空音频给出清晰错误。
* 不修改原始数据集。

标签映射必须集中定义，避免多处硬编码：

```python
TURN_LABELS = {
    0: "Complete",
    1: "InComplete",
    2: "Interrupt",
    3: "Backchannel",
}
```

文本标记映射：

| 标记              | 类别 |
| --------------- | -: |
| `<COMPLETE>`    |  0 |
| `<INCOMPLETE>`  |  1 |
| `<INTERRUPT>`   |  2 |
| `<BACKCHANNEL>` |  3 |

---

## 6. 环境与依赖

激活环境：

```bash
conda activate qwen
```

缺包时再安装依赖：

```bash
pip install "transformers>=4.30.0"
pip install "bitsandbytes>=0.41.0"
pip install peft accelerate
pip install "deepspeed>=0.10.0"
pip install qwen-omni-utils
pip install soundfile librosa
```

依赖规则：

* 优先使用当前 `qwen` conda 环境。
* 不使用 `sudo pip install`。
* 不随意升级 PyTorch、CUDA、bitsandbytes、DeepSpeed。
* 如需改依赖，必须说明原因和风险。
* 版本兼容问题先做最小复现，不直接大范围重装环境。

---

## 7. 常用命令

所有命令默认在 `/data1/wgy/qwen` 下执行。

原模型 smoke test：

```bash
python test_qwen3_omni.py
```

Turn Head 维度测试：

```bash
python - <<'PY'
import torch
from turn_head import TurnHead

head = TurnHead(hidden_size=2048, num_labels=4)
x = torch.randn(2, 100, 2048)
out = head(x)

assert out.shape == (2, 100, 4)
print("Turn Head shape test passed")
PY
```

单卡调试训练：

```bash
python train_turn.py \
  --model_path ./Qwen3-Omni-30B-A3B-Instruct \
  --dataset_path ./dataset/Easy-Turn \
  --output_dir ./output/turn_head_debug \
  --load_in_4bit \
  --lora_rank 8 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --max_steps 10
```

8 卡正式训练：

```bash
deepspeed --num_gpus=8 train_turn.py \
  --deepspeed ds_config.json \
  --model_path ./Qwen3-Omni-30B-A3B-Instruct \
  --dataset_path ./dataset/Easy-Turn \
  --output_dir ./output/turn_head \
  --load_in_4bit \
  --bnb_4bit_quant_type nf4 \
  --lora_rank 16 \
  --num_train_epochs 3 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 16
```

单音频推理测试：

```bash
python test_turn.py \
  --model_path ./output/turn_head/checkpoint-1000 \
  --audio_path ./dataset/Easy-Turn/Testset/complete/real/complete_real_001.wav
```

---

## 8. 代码契约

### `turn_head.py`

必须提供 `TurnHead`：

```python
class TurnHead(nn.Module):
    def __init__(self, hidden_size: int = 2048, num_labels: int = 4, ...):
        ...

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ...
```

要求：

* 输入：`[batch, seq_len, hidden_size]`
* 输出：`[batch, seq_len, num_labels]`
* 不在 `forward()` 中做文件 IO。
* 不强制 `.cuda()`；设备跟随输入 tensor。
* dtype 兼容 BF16、FP16 和量化模型输出。

### `turn_dataset.py`

要求：

* 兼容 Easy-Turn 目录结构。
* 统一解析标签。
* 支持训练集和测试集。
* 错误信息包含样本 key、路径和标签信息。
* 不修改原始数据。

### `uaf_model.py`

要求：

* 接收或加载 Qwen3-Omni 基座模型。
* 接入 Turn Head。
* 训练时可返回 `loss`。
* 推理时返回 `turn_logits`。
* 保持原模型推理能力不被破坏。
* 不混入 dataset、processor、trainer 逻辑。

### `train_turn.py`

要求：

* 支持 CLI 参数覆盖核心配置。
* 支持单卡调试和 DeepSpeed 多卡训练。
* 支持 4-bit NF4 加载。
* 支持 LoRA rank、学习率、batch size、epoch、max_steps 配置。
* 保存 LoRA adapter、Turn Head 权重、训练配置和标签映射。
* 记录 loss、学习率、显存/吞吐等关键信息。

### `test_turn.py`

要求：

* 支持单音频推理。
* 支持加载 checkpoint。
* 输出预测标签、概率和置信度。
* 不依赖训练进程中的全局状态。

---

## 9. Python 与 PyTorch 规范

Python：

* 使用类型注解。
* 路径使用 `pathlib.Path`。
* 日志优先使用 `logging`，避免大量裸 `print`。
* 不在 import 阶段加载大模型或读取大数据。
* 可调参数集中放在 `uaf_config.py` 或 argparse 中。
* 重要路径必须支持 CLI 覆盖。

PyTorch：

* 新 tensor 尽量从输入推断 `device` 和 `dtype`。
* 训练/推理模式明确使用 `model.train()` / `model.eval()`。
* 推理必须使用 `torch.no_grad()` 或 `torch.inference_mode()`。
* OOM 时优先降低 batch size、max sequence length 或 LoRA rank。
* checkpoint 必须能独立恢复推理。

---

## 10. 验收标准

| 修改类型          | 最小验证                  |
| ------------- | --------------------- |
| Turn Head 结构  | 运行 shape 测试           |
| Dataset 逻辑    | 读取少量样本并打印 key、标签、音频路径 |
| Model wrapper | 前向传播返回 `turn_logits`  |
| Training 脚本   | 单卡 `--max_steps 10`   |
| DeepSpeed 配置  | 多卡短步数 dry run         |
| Inference 脚本  | 单音频输出标签和概率            |

Claude 完成代码任务后，回复必须包含：

1. 修改文件列表。
2. 每个文件的关键变更。
3. 已运行命令和结果。
4. 未运行命令及原因。
5. 风险点和下一步建议。

---

## 11. 显存与 OOM 处理

| 方案          |     预估显存 | 结论          |
| ----------- | -------: | ----------- |
| BF16 全量加载   |   约 65GB | 单张 3090 不可行 |
| INT8 + LoRA | 约 17.8GB | 可行但余量较小     |
| QLoRA NF4   |  约 8.8GB | 默认推荐        |

OOM 处理顺序：

1. 降低 `per_device_train_batch_size`
2. 增大 `gradient_accumulation_steps`
3. 降低 `max_seq_len`
4. 启用 gradient checkpointing
5. 降低 LoRA rank
6. 检查是否误加载 Talker 或全量参数

---

## 12. 禁止事项

未经用户明确确认，不得执行：

```bash
rm -rf Qwen3-Omni-30B-A3B-Instruct
rm -rf dataset/Easy-Turn
rm -rf output/turn_head
git push
git commit
pip install --upgrade torch
conda remove
```

不得写入仓库：

* API key、token、密码
* 私有下载链接
* 个人账号
* 远程服务器凭据
* 大模型权重副本
* 原始数据集副本

---

## 13. 记忆维护

长期有效信息按以下规则沉淀：

| 信息类型         | 写入位置                 |
| ------------ | -------------------- |
| 团队共享项目规则     | `CLAUDE.md`          |
| 个人本地环境和偏好    | `CLAUDE.local.md`    |
| 调试经验、踩坑、命令修正 | Auto memory          |
| 特定目录/文件规则    | `.claude/rules/*.md` |
| 一次性任务细节      | 不保存                  |

建议定期用 `/memory` 检查重复、冲突和过期记忆。

---

## 14. 当前实施优先级

1. 实现并验证 `TurnHead`
2. 实现 Easy-Turn 数据加载
3. 实现 Qwen3-Omni + Turn Head wrapper
4. 跑通单卡 `max_steps=10` 调试训练
5. 跑通 8 卡 DeepSpeed dry run
6. 正式训练并保存 LoRA + Turn Head 权重
7. 实现单音频推理与测试集评估

优先围绕以上路径推进，避免无关重构。
