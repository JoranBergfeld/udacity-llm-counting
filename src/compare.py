"""
Compare the untuned base model ("OLD") against the LoRA-tuned model ("NEW").

Usage:
- python src/compare.py                 
- python src/compare.py --question "What is the capital of the Philippines?"
- python src/compare.py --no-lora
"""

import argparse
import os

from config import LORA_SAVE_DIR, SYSTEM_PROMPT_FORMAT, USE_UNSLOTH
from dataset import build_dataset
from model_setup import (
    disable_adapter,
    enable_adapter,
    generate,
    load_adapter,
    load_base_model,
)


def compare(model, tokenizer, messages, use_lora: bool) -> None:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if USE_UNSLOTH:
        old = generate(model, tokenizer, text, lora_request=None)
    else:
        disable_adapter(model)
        old = generate(model, tokenizer, text)
        enable_adapter(model)

    print("===OLD===\n")
    print(old)

    if not use_lora:
        return

    if not os.path.isdir(LORA_SAVE_DIR):
        print(f"\n(no adapter at {LORA_SAVE_DIR}/ — run src/train.py first, or pass --no-lora)")
        return

    lora_request = load_adapter(model, LORA_SAVE_DIR)
    new = generate(model, tokenizer, text, lora_request=lora_request)

    print("\n\n===NEW===\n")
    print(new)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, default=None,
                        help="Custom user question. Default: uses ds[0] from the letter-counting dataset.")
    parser.add_argument("--no-lora", action="store_true",
                        help="Only run the base model (skip the NEW tuned output).")
    args = parser.parse_args()

    model, tokenizer = load_base_model()

    if args.question:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_FORMAT},
            {"role": "user", "content": args.question},
        ]
    else:
        ds = build_dataset()
        messages = ds[0]["prompt"]
        print(f"Using ds[0]: word={ds[0]['words']!r} letter={ds[0]['letters']!r} count={ds[0]['counts']}")

    compare(model, tokenizer, messages, use_lora=not args.no_lora)
