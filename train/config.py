"""
UAF 双头训练配置
"""

SYSTEM_PROMPT = (
    "你是一个全双工语音助手的语音前端理解模块。\n\n"
    "请根据用户语音完成两个任务：\n"
    "1. 生成用户语音的文本转写。\n"
    "2. 判断当前用户语音的轮次状态。\n\n"
    "轮次状态定义如下：\n"
    "COMPLETE：用户已经说完，系统可以开始回答。\n"
    "INCOMPLETE：用户还没有说完，系统应该继续听。\n"
    "BACKCHANNEL：用户只是附和、回应或简短反馈，不应该打断系统。\n"
    "WAIT：用户要求暂停、停止、静音或终止当前交流。\n\n"
    "文本输出只包含用户语音的转写内容，不要输出 COMPLETE、INCOMPLETE、BACKCHANNEL、WAIT 等状态标签。\n"
    "轮次状态由专用分类头判断。"
)

LABEL2ID = {
    "<COMPLETE>": 0,
    "<INCOMPLETE>": 1,
    "<BACKCHANNEL>": 2,
    "<WAIT>": 3,
}

ID2STATE = {
    0: "<Complete>",
    1: "<InComplete>",
    2: "<Backchannel>",
    3: "<Interrupt>",
}

# 使用 weighted sampler 时，损失权重设为均匀
CLASS_WEIGHTS = [1.0, 1.0, 1.0, 1.0]

# 如果不用 sampler，可以用以下权重
# CLASS_WEIGHTS = [1.3, 1.0, 4.0, 4.0]

# Audio token IDs (Qwen3-Omni 特殊 token)
AUDIO_START_TOKEN_ID = 151669  # <|audio_start|>
AUDIO_END_TOKEN_ID = 151670    # <|audio_end|>
AUDIO_PAD_TOKEN_ID = 151675    # <|audio_pad|>
