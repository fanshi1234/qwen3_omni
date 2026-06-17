#!/usr/bin/env python3
"""
Qwen3-Omni 简单测试脚本
用于验证模型是否可以正常加载和推理
"""

import os
import sys
import torch
import soundfile as sf
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

# 模型路径
MODEL_PATH = "./Qwen3-Omni-30B-A3B-Instruct"

def main():
    print("=" * 50)
    print("Qwen3-Omni 测试脚本")
    print("=" * 50)

    # 检查模型目录是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型目录不存在 {MODEL_PATH}")
        print("请先下载模型: modelscope download --model Qwen/Qwen3-Omni-30B-A3B-Instruct --local_dir ./Qwen3-Omni-30B-A3B-Instruct")
        return

    # 检查是否有 safetensors 文件
    safetensors_files = [f for f in os.listdir(MODEL_PATH) if f.endswith('.safetensors')]
    if len(safetensors_files) < 15:
        print(f"警告: 模型文件不完整 (找到 {len(safetensors_files)}/15 个 safetensors 文件)")
        print("请等待模型下载完成")
        return

    print(f"\n1. 加载模型从: {MODEL_PATH}")
    print(f"   GPU数量: {torch.cuda.device_count()}")
    print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    try:
        # 加载模型
        print("\n2. 正在加载模型...")
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            dtype="auto",
            device_map="auto",
        )
        print("   模型加载成功!")

        # 加载处理器
        print("\n3. 正在加载处理器...")
        processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
        print("   处理器加载成功!")

        # 测试简单文本推理
        print("\n4. 测试文本推理...")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "你好！请用中文简单介绍一下你自己。"}
                ],
            },
        ]

        # 准备输入
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, return_tensors="pt", padding=True)
        inputs = inputs.to(model.device).to(model.dtype)

        # 生成回复
        print("   正在生成回复...")
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=200)

        # 解码回复
        # output_ids 是一个 tuple，第一个元素是 sequences
        if isinstance(output_ids, tuple):
            sequences = output_ids[0]
        else:
            sequences = output_ids

        response = processor.batch_decode(sequences[:, inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0]

        print("\n" + "=" * 50)
        print("模型回复:")
        print("=" * 50)
        print(response)
        print("=" * 50)

        print("\n✅ 测试完成！模型可以正常运行。")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
