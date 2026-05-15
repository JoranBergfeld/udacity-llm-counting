"""
Letter-counting dataset construction.

Mirrors notebook Cells 11-13. 
`build_dataset()` returns a HuggingFace Dataset with (words, letters, counts, prompt) columns ready for GRPOTrainer.
"""

import random

from config import SYSTEM_PROMPT_FORMAT

ALL_WORDS = [
    "idea", "glow", "rust", "maze", "echo", "wisp", "veto", "lush", "gaze",
    "knit", "fume", "plow", "void", "oath", "grim", "crisp", "lunar", "fable",
    "quest", "verge", "brawn", "elude", "aisle", "ember", "crave", "ivory",
    "mirth", "knack", "wryly", "onset", "mosaic", "velvet", "sphinx", "radius",
    "summit", "banner", "cipher", "glisten", "mantle", "scarab", "expose",
    "fathom", "tavern", "fusion", "relish", "lantern", "enchant", "torrent",
    "capture", "orchard", "eclipse", "frescos", "triumph", "absolve", "gossipy",
    "prelude", "whistle", "resolve", "zealous", "mirage", "aperture", "sapphire",
]


def generate_records():
    """Yield {words, letters, counts} triples, one per (word, in-word letter)
    plus a small number of (word, out-of-word letter) triples with count 0."""
    for word in ALL_WORDS:
        for letter in sorted(set(word)):
            yield {"words": word, "letters": letter, "counts": word.count(letter)}

        num_letters_not_in_word_left = int(len(word) // 7 + 1)
        random.seed(hash(word))
        all_letters = list("abcdefghijklmnopqrstuvwxyz")
        random.shuffle(all_letters)
        for letter in all_letters:
            if letter not in word:
                yield {"words": word, "letters": letter, "counts": 0}
                num_letters_not_in_word_left -= 1
            if num_letters_not_in_word_left == 0:
                break


def build_dataset(system_prompt: str = SYSTEM_PROMPT_FORMAT):
    from datasets import Dataset

    ds = Dataset.from_generator(generate_records)
    ds = ds.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": 'How many of the letter "{}" are there in the word "{}"'.format(
                        x["letters"], x["words"]
                    ),
                },
            ],
        }
    )
    return ds


if __name__ == "__main__":
    records = list(generate_records())
    print(f"generated {len(records)} records from {len(ALL_WORDS)} words")
    print("first record:", records[0])
    print("last record:", records[-1])

    try:
        ds = build_dataset()
        print(f"\ndataset length: {len(ds)}")
        print("sample[0]:", ds[0])
    except ImportError as e:
        print(f"\n(skipping build_dataset(): {e})")
