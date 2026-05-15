"""
GRPO training for the letter-counting task.

Usage:
python src/train.py --max-steps 5     # quick smoke (~2-5 min)
python src/train.py --max-steps 100   # full run (~30-60 min)

Produces:
grpo_saved_lora/adapter_model.safetensors
outputs/reward_log.csv
outputs/reward_plot.png
"""

import argparse
import os

from config import (
    COMMON_GRPO_PARAMS,
    LORA_SAVE_DIR,
    REWARD_LOG_PATH,
    REWARD_PLOT_PATH,
    USE_UNSLOTH,
)
from dataset import build_dataset
from model_setup import load_base_model
from rewards import REWARD_FUNCTIONS


def main(max_steps: int) -> None:
    from trl import GRPOConfig, GRPOTrainer

    model, tokenizer = load_base_model()
    ds = build_dataset()

    training_args = GRPOConfig(
        **COMMON_GRPO_PARAMS, max_steps=max_steps, use_vllm=USE_UNSLOTH
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=REWARD_FUNCTIONS,
        args=training_args,
        train_dataset=ds,
    )
    trainer.train()

    if USE_UNSLOTH:
        model.save_lora(LORA_SAVE_DIR)
    else:
        model.save_pretrained(LORA_SAVE_DIR)
    print(f"saved LoRA adapter to {LORA_SAVE_DIR}/")

    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(REWARD_LOG_PATH), exist_ok=True)
    log_df = pd.DataFrame(trainer.state.log_history)
    log_df.to_csv(REWARD_LOG_PATH, index=False)
    print(f"wrote reward log to {REWARD_LOG_PATH}")

    fig, ax = plt.subplots()
    if "reward" in log_df.columns:
        log_df["reward"].plot(ax=ax, label="reward")
    if "rewards/correct_answer_reward_func/mean" in log_df.columns:
        log_df["rewards/correct_answer_reward_func/mean"].plot(
            ax=ax, label="rewards/correct_answer_reward_func/mean"
        )
    ax.legend()
    ax.set_xlabel("step")
    fig.savefig(REWARD_PLOT_PATH)
    print(f"wrote reward plot to {REWARD_PLOT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=5)
    args = parser.parse_args()
    main(args.max_steps)
