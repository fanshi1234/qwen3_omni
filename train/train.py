"""
UAF 双头训练

序列: [system_prompt] [audio] [text_target]
LM Head: 只在 text_target 上计算 loss
TD Head: 只使用 audio hidden, attention pooling → [B, 4]
"""

import os
import argparse
import logging
import wave
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from config import (
    SYSTEM_PROMPT, LABEL2ID, ID2STATE,
    CLASS_WEIGHTS, AUDIO_START_TOKEN_ID, AUDIO_END_TOKEN_ID,
)
from dataset import TurnDataset, collate_fn, get_weighted_sampler, get_balanced_sampler
from td_head import TDHead

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_metrics(labels, preds):
    """计算分类指标"""
    # 保护：空数据返回默认值
    if len(labels) == 0 or len(preds) == 0:
        return {
            "acc": 0.0,
            "macro_f1": 0.0,
            "per_class_acc": [0.0, 0.0, 0.0, 0.0],
            "per_class_f1": [0.0, 0.0, 0.0, 0.0],
            "per_class_precision": [0.0, 0.0, 0.0, 0.0],
            "per_class_recall": [0.0, 0.0, 0.0, 0.0],
            "label_names": ["Complete", "InComplete", "Backchannel", "Wait"],
        }

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
    per_class_precision = precision_score(labels, preds, average=None, zero_division=0)
    per_class_recall = recall_score(labels, preds, average=None, zero_division=0)

    # 每个类别的 accuracy
    label_names = ["Complete", "InComplete", "Backchannel", "Wait"]
    per_class_acc = []
    for cls_id in range(4):
        cls_mask = np.array(labels) == cls_id
        if cls_mask.sum() > 0:
            cls_acc = (np.array(preds)[cls_mask] == cls_id).mean()
            per_class_acc.append(cls_acc)
        else:
            per_class_acc.append(0.0)

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "per_class_acc": per_class_acc,
        "per_class_f1": per_class_f1,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "label_names": label_names,
    }


def log_metrics(logger, prefix, metrics):
    """输出指标日志"""
    logger.info(f"  [{prefix}] Acc={metrics['acc']:.4f} | Macro-F1={metrics['macro_f1']:.4f}")
    for i, name in enumerate(metrics['label_names']):
        # 安全获取指标，避免索引越界
        acc = metrics['per_class_acc'][i] if i < len(metrics['per_class_acc']) else 0.0
        p = metrics['per_class_precision'][i] if i < len(metrics['per_class_precision']) else 0.0
        r = metrics['per_class_recall'][i] if i < len(metrics['per_class_recall']) else 0.0
        f1 = metrics['per_class_f1'][i] if i < len(metrics['per_class_f1']) else 0.0
        logger.info(f"    {name:>12}: Acc={acc:.4f} | P={p:.4f} | R={r:.4f} | F1={f1:.4f}")


class UAFModel(torch.nn.Module):
    def __init__(self, model_path, training_stage="A", use_lora=False, lora_rank=8, lora_alpha=16):
        super().__init__()

        self.training_stage = training_stage
        self.use_lora = use_lora

        logger.info("=" * 60)
        logger.info("Initializing UAF Model with QLoRA")
        logger.info("=" * 60)

        # ========== 1. FP16 加载 (分布到 8 卡) ==========
        logger.info("Step 1: Loading model with FP16 + device_map=auto ...")

        # 使用 device_map="auto" 让模型自动分布到多张卡
        # 不使用量化，直接 FP16
        # 注意: 使用 accelerate 的 dispatch_model 来正确处理多设备
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map

        self.base_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",  # 自动分布到 8 张卡
            low_cpu_mem_usage=True,
        )
        logger.info(f"Model distributed across GPUs")

        # 检查量化后的模型大小
        model_size = sum(p.numel() * p.element_size() for p in self.base_model.parameters()) / 1024**3
        logger.info(f"   Quantized model size: {model_size:.2f} GB")

        # ========== 2. 冻结基座模型 ==========
        logger.info("Step 2: Freezing base model ...")
        self.base_model.eval()
        for p in self.base_model.parameters():
            p.requires_grad = False

        # ========== 3. 添加 LoRA (QLoRA) ==========
        if use_lora:
            logger.info(f"Step 3: Adding LoRA adapters (rank={lora_rank}, alpha={lora_alpha}) ...")
            from peft import LoraConfig, get_peft_model, TaskType

            # 注意: FP16 + device_map="auto" 不需要 prepare_model_for_kbit_training
            # 该函数只用于 bitsandbytes 量化模型

            # LoRA 配置 - 只应用到 Attention 层 (避免 MoE 兼容问题)
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                bias="none",
                target_modules=[
                    # Attention 层
                    "q_proj", "k_proj", "v_proj", "o_proj",
                ],
                layers_to_transform=list(range(32, 48)),  # 后 16 层
            )

            # 应用 LoRA
            self.base_model = get_peft_model(self.base_model, peft_config)

            # 打印可训练参数
            trainable_params = sum(p.numel() for p in self.base_model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.base_model.parameters())
            logger.info(f"   Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.4f}%)")
            logger.info(f"   Total params: {total_params:,}")
        else:
            logger.info("Step 3: Skipping LoRA (Stage A - only TD Head)")

        # ========== 4. 添加 TD Head ==========
        logger.info("Step 4: Adding TD Head ...")
        hidden_size = self.base_model.config.thinker_config.text_config.hidden_size
        self.td_head = TDHead(hidden_size=hidden_size, num_labels=4)

        td_params = sum(p.numel() for p in self.td_head.parameters())
        logger.info(f"   TD Head params: {td_params:,}")

        # ========== 5. 加载 Processor ==========
        logger.info("Step 5: Loading processor ...")
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)

        # ========== 完成 ==========
        logger.info("=" * 60)
        logger.info("✅ UAF Model initialized successfully")
        logger.info(f"   Training stage: {training_stage}")
        logger.info(f"   LoRA: {'enabled' if use_lora else 'disabled'}")
        logger.info(f"   TD Head: enabled")
        logger.info("=" * 60)

    def _load_audio(self, wav_path):
        """加载音频文件并重采样到 16kHz"""
        try:
            with wave.open(wav_path, 'r') as wf:
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                sr = wf.getframerate()

            # 重采样到 16kHz
            if sr != 16000:
                try:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                    sr = 16000
                except Exception:
                    pass

            return audio, sr
        except Exception:
            return np.zeros(16000, dtype=np.float32), 16000

    def _prepare_inputs(self, audio_list, text_targets):
        """准备模型输入"""
        batch_size = len(audio_list)

        all_input_ids = []
        all_labels = []
        all_audio_positions = []

        for i in range(batch_size):
            # 加载音频
            audio_np, sr = self._load_audio(audio_list[i])

            # 构建消息
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "audio", "audio": audio_np},
                ]},
                {"role": "assistant", "content": text_targets[i]},
            ]

            # Tokenize
            text = self.processor.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False
            )
            inputs = self.processor(
                text=text,
                audio=[audio_np],
                sampling_rate=sr,
                return_tensors="pt",
                padding=False,
            )

            input_ids = inputs["input_ids"][0]
            all_input_ids.append(input_ids)

            # 找 audio 位置
            audio_start = (input_ids == AUDIO_START_TOKEN_ID).nonzero(as_tuple=True)[0]
            audio_end = (input_ids == AUDIO_END_TOKEN_ID).nonzero(as_tuple=True)[0]
            if len(audio_start) > 0 and len(audio_end) > 0:
                all_audio_positions.append((audio_start[0].item(), audio_end[0].item()))
            else:
                logger.warning(f"Audio tokens not found in sample {i}")
                all_audio_positions.append((0, 0))

            # Labels: 只有 text_target 部分有值
            target_ids = self.processor.tokenizer(
                text_targets[i], return_tensors="pt", padding=False
            ).input_ids[0]

            labels = torch.full_like(input_ids, -100)
            target_len = len(target_ids)
            if target_ids[-1] == self.processor.tokenizer.eos_token_id:
                target_len -= 1
            labels[-target_len:] = input_ids[-target_len:]
            all_labels.append(labels)

        # Padding - 保持在 CPU 上，让 transformers 自动处理设备转换
        max_len = max(ids.size(0) for ids in all_input_ids)
        padded_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        padded_labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        padded_mask = torch.zeros(batch_size, max_len, dtype=torch.long)

        for i in range(batch_size):
            L = all_input_ids[i].size(0)
            padded_ids[i, :L] = all_input_ids[i]
            padded_labels[i, :L] = all_labels[i]
            padded_mask[i, :L] = 1

        return {
            "input_ids": padded_ids,
            "attention_mask": padded_mask,
            "labels": padded_labels,
            "audio_positions": all_audio_positions,
        }

    def forward(self, audio_list, text_targets, turn_labels=None):
        """前向传播"""
        inputs = self._prepare_inputs(audio_list, text_targets)

        # 获取设备
        device = next(self.base_model.parameters()).device

        # 将 td_head 移到正确设备（只在需要时）
        if next(self.td_head.parameters()).device != device:
            self.td_head = self.td_head.to(device)

        # Thinker forward
        outputs = self.base_model.thinker(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states[-1]
        batch_size = hidden_states.size(0)

        # 提取 audio hidden (只取 <|audio_start|> 和 <|audio_end|> 之间的 tokens)
        audio_hiddens = []
        audio_masks = []
        for i in range(batch_size):
            s, e = inputs["audio_positions"][i]
            if e > s + 1:
                ah = hidden_states[i, s+1:e, :]
                am = torch.ones(ah.size(0), device=device)
            else:
                ah = hidden_states[i, :1, :]
                am = torch.ones(1, device=device)
            audio_hiddens.append(ah)
            audio_masks.append(am)

        # Padding audio hidden
        max_a = max(ah.size(0) for ah in audio_hiddens)
        H = hidden_states.size(-1)
        padded_audio = torch.zeros(batch_size, max_a, H, device=device)
        padded_amask = torch.zeros(batch_size, max_a, device=device)
        for i in range(batch_size):
            A = audio_hiddens[i].size(0)
            padded_audio[i, :A] = audio_hiddens[i]
            padded_amask[i, :A] = audio_masks[i]

        # TD Head
        turn_logits = self.td_head(padded_audio, padded_amask)

        result = {"turn_logits": turn_logits}

        # 计算损失
        if turn_labels is not None:
            # TD loss
            cw = torch.tensor(CLASS_WEIGHTS, device=turn_logits.device, dtype=torch.float32)
            loss_td = F.cross_entropy(turn_logits.float(), turn_labels.to(turn_logits.device), weight=cw)
            result["loss_td"] = loss_td

            # Stage A: 只训练 TD Head
            # Stage B/C: 同时训练 TD + ASR
            if self.training_stage in ["B", "C"]:
                lm_logits = self.base_model.thinker.lm_head(hidden_states)
                labels_on_device = inputs["labels"].to(lm_logits.device)
                loss_text = F.cross_entropy(
                    lm_logits.float().reshape(-1, lm_logits.size(-1)),
                    labels_on_device.reshape(-1),
                    ignore_index=-100,
                )
                result["loss_text"] = loss_text
                # 确保两个 loss 在同一设备
                result["loss"] = loss_td.to(loss_text.device) + 0.1 * loss_text
            else:
                result["loss_text"] = torch.tensor(0.0, device=loss_td.device)
                result["loss"] = loss_td

        return result


def evaluate_on_testset(model, eval_list_file, evalset_dir, max_audio_seconds, batch_size=4):
    """在 Testset 上评估模型"""
    eval_dataset = TurnDataset(
        list_file=eval_list_file,
        trainset_dir=evalset_dir,
        max_duration=max_audio_seconds,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", ncols=80):
            outputs = model(
                audio_list=batch["audio_list"],
                text_targets=batch["text_targets"],
            )
            preds = torch.argmax(outputs["turn_logits"], dim=-1).cpu().numpy()
            labels = batch["turn_labels"].numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    return compute_metrics(all_labels, all_preds)


def train(args):
    # 初始化模型
    model = UAFModel(
        model_path=args.model_path,
        training_stage=args.stage,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # 加载第一阶段 TD Head 权重
    if args.load_td_head:
        if os.path.exists(args.load_td_head):
            logger.info(f"Loading Stage A TD Head weights from {args.load_td_head}")
            model.td_head.load_state_dict(torch.load(args.load_td_head, map_location="cpu"))
            logger.info("Stage A TD Head weights loaded successfully")
        else:
            logger.error(f"TD Head weights not found at {args.load_td_head}")
            return

    # 加载数据集
    train_dataset = TurnDataset(
        list_file=args.train_list_file,
        trainset_dir=args.trainset_dir,
        max_duration=args.max_audio_seconds,
        max_samples=args.max_samples,
    )

    # 采样策略
    if args.sampler == "weighted":
        sampler = get_weighted_sampler(train_dataset)
        shuffle = False
    elif args.sampler == "balanced":
        sampler = get_balanced_sampler(train_dataset)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=2,
        collate_fn=collate_fn,
    )

    # 参数组配置
    optimizer_grouped_parameters = []

    # TD Head 参数
    td_head_params = [p for p in model.td_head.parameters() if p.requires_grad]
    if td_head_params:
        optimizer_grouped_parameters.append({
            "params": td_head_params,
            "lr": args.lr_td_head,
            "weight_decay": 0.01,
        })
        logger.info(f"TD Head params: {sum(p.numel() for p in td_head_params) / 1e6:.4f}M")

    # LoRA 参数
    if args.use_lora:
        lora_params = [p for n, p in model.base_model.named_parameters()
                       if p.requires_grad and 'lora' in n]
        if lora_params:
            optimizer_grouped_parameters.append({
                "params": lora_params,
                "lr": args.lr_lora,
                "weight_decay": 0.01,
            })
            logger.info(f"LoRA params: {sum(p.numel() for p in lora_params) / 1e6:.4f}M")

    # Audio Projector 参数 (Stage B/C)
    if args.stage in ["B", "C"]:
        audio_proj_params = []
        for n, p in model.base_model.named_parameters():
            if p.requires_grad and ('audio_projector' in n or 'audio_tower' in n):
                audio_proj_params.append(p)
        if audio_proj_params:
            optimizer_grouped_parameters.append({
                "params": audio_proj_params,
                "lr": args.lr_audio_projector,
                "weight_decay": 0.01,
            })
            logger.info(f"Audio Projector params: {sum(p.numel() for p in audio_proj_params) / 1e6:.4f}M")

    total_trainable = sum(p.numel() for group in optimizer_grouped_parameters for p in group["params"])
    logger.info(f"Total trainable params: {total_trainable / 1e6:.4f}M")

    # 优化器配置
    # NF4 量化模型 + DeepSpeed 有兼容性问题，使用普通优化器
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    model_engine = model
    logger.info("Using standard optimizer (NF4 + DeepSpeed not compatible)")

    # 早停配置
    early_stopping_patience = args.early_stopping_patience
    best_metric = 0.0
    patience_counter = 0
    best_checkpoint_path = None

    # 训练配置日志
    logger.info("=" * 60)
    logger.info("Starting training")
    logger.info(f"  Stage: {args.stage}")
    logger.info(f"  Epochs: {args.num_train_epochs}")
    logger.info(f"  Batch: {args.per_device_train_batch_size}")
    logger.info(f"  Grad accum: {args.gradient_accumulation_steps}")
    logger.info(f"  Sampler: {args.sampler}")
    logger.info(f"  DeepSpeed: {args.deepspeed}")
    logger.info(f"  Early stopping: {early_stopping_patience} evals")
    logger.info("=" * 60)

    # 判断是否为 rank0 (只在 rank0 打印日志)
    is_rank0 = True
    if hasattr(args, 'local_rank') and args.local_rank != -1:
        is_rank0 = args.local_rank == 0
    elif hasattr(args, 'global_rank'):
        is_rank0 = args.global_rank == 0

    global_step = 0
    model_engine.train()
    all_preds = []
    all_labels = []

    for epoch in range(args.num_train_epochs):
        if is_rank0:
            logger.info(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")

        # 只在 rank0 显示进度条
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.num_train_epochs}",
            ncols=100,
            leave=True,
            disable=not is_rank0,
            mininterval=5,      # 最小更新间隔 5 秒
            miniters=50,        # 最小更新步数 50
        )

        for batch_idx, batch in enumerate(pbar):
            # 获取设备
            if args.deepspeed:
                device = model_engine.device
            else:
                device = next(model_engine.base_model.parameters()).device

            # Forward
            outputs = model_engine(
                audio_list=batch["audio_list"],
                text_targets=batch["text_targets"],
                turn_labels=batch["turn_labels"].to(device),
            )

            loss = outputs["loss"]

            # Backward
            if args.deepspeed:
                model_engine.backward(loss)
                model_engine.step()
            else:
                loss.backward()
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for group in optimizer_grouped_parameters for p in group["params"]],
                        1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad()

            # 收集预测
            preds = torch.argmax(outputs["turn_logits"], dim=-1).cpu().numpy()
            labels = batch["turn_labels"].numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

            # 梯度累积计数
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1

                # 日志 - 每 100 步更新一次进度条
                if global_step % 100 == 0 and is_rank0:
                    metrics = compute_metrics(all_labels, all_preds)

                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'acc': f"{metrics['acc']:.4f}",
                        'f1': f"{metrics['macro_f1']:.4f}",
                    })

                    all_preds = []
                    all_labels = []

                # 详细日志 - 每 logging_steps 步输出
                if global_step % args.logging_steps == 0 and is_rank0:
                    metrics = compute_metrics(all_labels, all_preds)
                    logger.info(f"Step {global_step} | Loss: {loss.item():.4f} | Acc: {metrics['acc']:.4f} | F1: {metrics['macro_f1']:.4f}")
                    all_preds = []
                    all_labels = []

                # 保存 checkpoint
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    if is_rank0:
                        os.makedirs(save_path, exist_ok=True)
                        if args.deepspeed:
                            model_engine.save_checkpoint(save_path)
                        else:
                            torch.save(model_engine.td_head.state_dict(), os.path.join(save_path, "td_head.pt"))
                        logger.info(f"Saved checkpoint to {save_path}")

                # Testset 评估
                if args.eval_list_file and global_step % args.eval_steps == 0:
                    if is_rank0:
                        logger.info(f"\n{'='*50}")
                        logger.info(f"Evaluating on Testset at step {global_step}...")

                    eval_metrics = evaluate_on_testset(
                        model_engine, args.eval_list_file, args.evalset_dir,
                        args.max_audio_seconds, batch_size=args.per_device_eval_batch_size,
                    )

                    if is_rank0:
                        log_metrics(logger, "Testset", eval_metrics)

                        # 早停检查
                        test_f1 = eval_metrics['macro_f1']
                        if test_f1 > best_metric:
                            best_metric = test_f1
                            patience_counter = 0
                            best_path = os.path.join(args.output_dir, "best_checkpoint")
                            os.makedirs(best_path, exist_ok=True)
                            if args.deepspeed:
                                model_engine.save_checkpoint(best_path)
                            else:
                                torch.save(model_engine.td_head.state_dict(), os.path.join(best_path, "td_head.pt"))
                            best_checkpoint_path = best_path
                            logger.info(f"🎯 New best Testset Macro-F1: {test_f1:.4f} -> Saved to {best_path}")
                        else:
                            patience_counter += 1
                            logger.info(f"No improvement for {patience_counter}/{early_stopping_patience} evals")
                            if patience_counter >= early_stopping_patience:
                                logger.info(f"🛑 Early stopping triggered at step {global_step}")
                                break

                    model_engine.train()

        # 检查早停
        if patience_counter >= early_stopping_patience:
            break

    # 保存最终模型
    final_path = os.path.join(args.output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    if args.deepspeed:
        model_engine.save_checkpoint(final_path)
    else:
        torch.save(model_engine.td_head.state_dict(), os.path.join(final_path, "td_head.pt"))

    # 最终评估
    if args.eval_list_file:
        logger.info(f"\n{'='*60}")
        logger.info("Final evaluation on Testset...")
        final_metrics = evaluate_on_testset(
            model_engine, args.eval_list_file, args.evalset_dir,
            args.max_audio_seconds, batch_size=args.per_device_eval_batch_size,
        )
        log_metrics(logger, "Final", final_metrics)

    # 汇总
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Training completed!")
    logger.info(f"   Best Testset Macro-F1: {best_metric:.4f}")
    if best_checkpoint_path:
        logger.info(f"   Best checkpoint: {best_checkpoint_path}")
    logger.info(f"   Final model: {final_path}")
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="UAF Turn Head Training")
    parser.add_argument("--model_path", type=str, default="./Qwen3-Omni-30B-A3B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./train/output")
    parser.add_argument("--train_list_file", type=str, required=True)
    parser.add_argument("--trainset_dir", type=str, default="./dataset/Easy-Turn/Trainset")
    parser.add_argument("--eval_list_file", type=str, default=None)
    parser.add_argument("--evalset_dir", type=str, default="./dataset/Easy-Turn/Testset")
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--sampler", type=str, default="weighted",
                        choices=["none", "weighted", "balanced"])
    parser.add_argument("--stage", type=str, default="A", choices=["A", "B", "C"])
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_audio_seconds", type=float, default=10.0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", type=str, default=None,
                        help="DeepSpeed config file path")

    # 学习率参数
    parser.add_argument("--lr_td_head", type=float, default=5e-4)
    parser.add_argument("--lr_lora", type=float, default=5e-5)
    parser.add_argument("--lr_audio_projector", type=float, default=5e-5)

    # 加载第一阶段权重
    parser.add_argument("--load_td_head", type=str, default=None,
                        help="Path to stage_a td_head.pt to load")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
