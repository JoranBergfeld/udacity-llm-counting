"""
Shared configuration constants.

All tunable knobs live here so the rest of the scripts stay importable without pulling in heavy deps. 

Backend selection:
- UNSLOTH=1: unsloth + vLLM fast path (Linux/WSL, Vocareum)
- UNSLOTH=0: transformers + PEFT fallback (portable, slower)
"""

import os

USE_UNSLOTH = os.environ.get("UNSLOTH", "1") != "0"

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_SEQ_LENGTH = 384
GPU_MEMORY_UTILIZATION = 0.5

LORA_RANK = 64
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

SYSTEM_PROMPT_COT = """
You count the occurrences of a letter in a word by spelling the word out letter-by-letter and maintaining a running total.

Respond in exactly this format:
<reasoning>
Counting the number of [letter]'s in the word [word]
1. [first letter of word] - [running total] so far
2. [second letter of word] - [running total] so far
...

The letter "[letter]" appears [final total] times in the word "[word]".
</reasoning>
<answer>
[final total]
</answer>

Example:
Question: How many of the letter "o" are there in the word "room"
<reasoning>
Counting the number of o's in the word room
1. r - 0 so far
2. o - 1 so far
3. o - 2 so far
4. m - 2 so far
The letter "o" appears 2 times in the word "room".
</reasoning>
<answer>
2
</answer>
"""

SYSTEM_PROMPT_FORMAT = """
Respond in the following format:
<reasoning>
Counting the number of [letter_to_count]'s in the word [word]
1. [first letter] - [count of requested letter so far] so far
2. [second letter] - [count of requested letter so far] so far
...
</reasoning>
<answer>
[number]
</answer>
"""

COMMON_GRPO_PARAMS = dict(
    learning_rate=1e-5,
    beta=1e-4,
    per_device_train_batch_size=16,
    num_generations=4,
    gradient_accumulation_steps=1,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    logging_steps=1,
    max_prompt_length=256,
    max_completion_length=200,
    num_train_epochs=1,
    save_steps=250,
    max_grad_norm=0.1,
    report_to="none",
    output_dir="outputs",
)
# use_vllm is intentionally NOT in COMMON_GRPO_PARAMS: it is environment-
# dependent, not a hyperparameter. Callers pass it explicitly (train.py derives
# it from USE_UNSLOTH; the notebook passes use_vllm=False with a written reason).

LORA_SAVE_DIR = "grpo_saved_lora"
REWARD_LOG_PATH = "outputs/reward_log.csv"
REWARD_PLOT_PATH = "outputs/reward_plot.png"
