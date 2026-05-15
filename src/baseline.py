"""
Prompt-engineering baseline: blank system prompt vs Chain-of-Thought prompt.

Mirrors notebook Cells 6 and 8. 
The CoT prompt includes one worked example('room' -> 2 o's) satisfying the rubric's single-example CoT requirement.
"""

from config import SYSTEM_PROMPT_COT
from model_setup import generate, load_base_model

USER_PROMPT = 'How many of the letter "g" are there in the word "engage"'


def run(model, tokenizer, system_prompt: str) -> str:
    text = tokenizer.apply_chat_template(
        conversation=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": USER_PROMPT},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return generate(model, tokenizer, text, max_tokens=2048)


if __name__ == "__main__":
    model, tokenizer = load_base_model()

    print("=" * 60)
    print("BLANK SYSTEM PROMPT")
    print("=" * 60)
    print(run(model, tokenizer, ""))

    print("\n" + "=" * 60)
    print("CoT SYSTEM PROMPT (single-example)")
    print("=" * 60)
    print(run(model, tokenizer, SYSTEM_PROMPT_COT))
