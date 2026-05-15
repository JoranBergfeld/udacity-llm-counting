"""
Reward functions for GRPO training on the letter-counting task.

Each function returns positive values for desired behavior and negative for undesired. 
The asserts at the bottom serve as rubric evidence. 
Running this file as __main__ is the validation artifact.
"""

import re
from collections import Counter


def extract_letter_numbering(response: str) -> list[int]:
    """Extract the leading integer from each '\\n<n>. <letter>' line."""
    pattern = r"\n(\d+)\. [a-z]"
    matches = re.findall(pattern, response, flags=re.IGNORECASE)
    return [int(m) for m in matches] if matches else []


def extract_spelling(response: str) -> str:
    """Concatenate the single letters from each '\\n<n>. <letter>' line."""
    pattern = r"\n\d+\. ([a-z])"
    matches = re.findall(pattern, response, flags=re.IGNORECASE)
    return "".join(matches) if matches else ""


def get_resp_letters_and_counts(response: str) -> list[tuple[str, str]]:
    """Parse lines of form '\\n<n>. <letter> - <count> so far' into (letter, count)."""
    pattern = r"\n(\d+)\. ([a-z])\D*(\d+)"
    matches = re.findall(pattern, response, flags=re.IGNORECASE)
    if not matches:
        return []
    return [(letter, count) for _, letter, count in matches]


def extract_xml_answer(text: str) -> str:
    """Return the string inside <answer>...</answer>, or empty string."""
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def numbering_reward_func(completions, words, **kwargs) -> list[float]:
    """+0.5 per in-order numbering, -0.5 per out-of-order, -1.0 per overrun, / len(word)."""
    responses = [c[0]["content"] for c in completions]
    res = []
    for response, word in zip(responses, words):
        reward = 0.0
        for ix, spell_number in enumerate(extract_letter_numbering(response)):
            line_number = ix + 1
            if spell_number == line_number:
                reward += 0.5
            else:
                reward -= 0.5
            if line_number > len(word):
                reward -= 1.0
        res.append(reward / len(word))
    return res


def spelling_reward_func(completions, words, **kwargs) -> list[float]:
    """+2.0 if exact spelling; -0.5 per length diff; -1.0 per extra letter; -0.5 per missing."""
    responses = [c[0]["content"] for c in completions]
    res = []
    for word, response in zip(words, responses):
        spelled = extract_spelling(response).lower()
        target = word.lower()
        reward = 0.0

        if spelled == target:
            reward += 2.0

        reward -= 0.5 * abs(len(spelled) - len(target))

        spelled_counts = Counter(spelled)
        target_counts = Counter(target)

        extra = sum((spelled_counts - target_counts).values())
        missing = sum((target_counts - spelled_counts).values())

        reward -= 1.0 * extra
        reward -= 0.5 * missing

        res.append(reward)
    return res


def counting_reward_func(completions, letters, **kwargs) -> list[float]:
    """+1.0 per correct running total, -1.0 per incorrect; normalized by line count. -1.0 if unparsable."""
    responses = [c[0]["content"] for c in completions]
    res = []
    for letter, response in zip(letters, responses):
        reward = 0.0
        pairs = get_resp_letters_and_counts(response)
        if not pairs:
            res.append(-1.0)
            continue

        actual_count = 0
        for resp_letter, resp_count in pairs:
            if letter.lower() == resp_letter.lower():
                actual_count += 1
            if int(resp_count) == actual_count:
                reward += 1.0
            else:
                reward -= 1.0

        res.append(reward / len(pairs))
    return res


def format_reward_func(completions, **kwargs) -> list[float]:
    """+0.5 if matches <reasoning>...</reasoning><answer>...</answer>; +0.5 if answer is a digit."""
    pattern = r"\s*<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    res = []
    for completion in completions:
        reward = 0.0
        response = completion[0]["content"]
        if re.match(pattern, response, flags=re.MULTILINE | re.DOTALL):
            reward += 0.5
        extracted = extract_xml_answer(response)
        if extracted.isdigit():
            reward += 0.5
        res.append(reward)
    return res


def correct_answer_reward_func(prompts, completions, counts, **kwargs) -> list[float]:
    """+2.0 for exact match on final count, -1.0 otherwise."""
    responses = [c[0]["content"] for c in completions]
    extracted = [extract_xml_answer(r) for r in responses]

    print(f"""
{"-" * 20}
Question: {prompts[0][-1]["content"]}
Answer: {counts[0]}
Response: {responses[0]}
Extracted: {extracted[0]}
Correct: {str(extracted[0]) == str(counts[0])}!
    """)

    return [2.0 if str(r) == str(a) else -1.0 for r, a in zip(extracted, counts)]


REWARD_FUNCTIONS = [
    numbering_reward_func,
    spelling_reward_func,
    counting_reward_func,
    format_reward_func,
    correct_answer_reward_func,
]


def assert_extractors() -> None:
    """Validate the four regex extractor helpers."""
    assert extract_letter_numbering(
        "\n1. g - 1 so far\n2. o - 1 so far\n3. a - 2 so far\n4. a - 2 so far\n5. l - 2 so far\n"
    ) == [1, 2, 3, 4, 5]
    print("PASS extract_letter_numbering")

    assert extract_spelling(
        "Here is a letter by letter spelling:\n\n1. g - 1 so far\n2. o - 1 so far\n3. a - 2 so far\n4. l - 2 so far\n5. l - 2 so far\n"
    ) == "goall"
    print("PASS extract_spelling")

    assert get_resp_letters_and_counts(
        "\n1. g - 1 so far\n2. o - 1 so far\n3. a - 2 so far\n4. a - 2 so far\n5. l - 2 so far\n"
    ) == [("g", "1"), ("o", "1"), ("a", "2"), ("a", "2"), ("l", "2")]
    print("PASS get_resp_letters_and_counts")

    assert (
        extract_xml_answer(
            "\n<reasoning>\nThis is my reasoning.\n</reasoning>\n<answer>SUPERCALIFRAGILISTICEXPIALIDOCIOUS</answer>\n"
        )
        == "SUPERCALIFRAGILISTICEXPIALIDOCIOUS"
    )
    print("PASS extract_xml_answer")


def assert_numbering_reward() -> None:
    """A worse-numbered completion must score below a better-numbered one."""
    res = numbering_reward_func(
        completions=[
            [{"content": """<reasoning>
Here is a letter by letter spelling:
1. g - 1 so far
2. o - 1 so far
3. a - 2 so far
3. l - 2 so far
1. l - 2 so far
1. l - 2 so far
</reasoning>
<answer>2</answer>"""}],
            [{"content": """<reasoning>
Here is a letter by letter spelling:
1. g - 1 so far
2. o - 1 so far
3. a - 2 so far
3. l - 2 so far
</reasoning>
<answer>2</answer>"""}],
        ],
        words=["goal", "goal"],
    )
    assert res[1] > res[0], f"numbering: {res}"
    print(f"PASS numbering_reward_func {res}")


def assert_spelling_reward() -> None:
    """A misspelled completion must score below a correctly spelled one."""
    res = spelling_reward_func(
        completions=[
            [{"content": """<reasoning>
Here is a letter by letter spelling:
1. g - 1 so far
2. o - 1 so far
3. a - 2 so far
4. l - 2 so far
5. l - 2 so far
</reasoning>
<answer>2</answer>"""}],
            [{"content": """<reasoning>
Here is a letter by letter spelling:
1. g - 1 so far
2. o - 1 so far
3. a - 2 so far
4. l - 2 so far
</reasoning>
<answer>2</answer>"""}],
        ],
        words=["goal", "goal"],
    )
    assert res[1] > res[0], f"spelling: {res}"
    print(f"PASS spelling_reward_func {res}")


def assert_counting_reward() -> None:
    """A completion with wrong running totals must score below a correct one."""
    res = counting_reward_func(
        completions=[
            [{"content": """<reasoning>
Here is a letter by letter spelling:

1. g - 0 so far
2. o - 0 so far
3. a - 1 so far
4. a - 2 so far
5. l - 0 so far

</reasoning>
<answer>
This is my answer.
</answer>"""}],
            [{"content": """<reasoning>
Here is a letter by letter spelling:

1. g - 1 so far
2. o - 1 so far
3. a - 1 so far
4. a - 1 so far
5. l - 1 so far

</reasoning>
<answer>
This is my answer.
</answer>"""}],
        ],
        letters=["g", "g"],
    )
    assert res[1] > res[0], f"counting: {res}"
    print(f"PASS counting_reward_func {res}")


def assert_format_reward() -> None:
    """A free-text completion must score below a correctly tagged one."""
    res = format_reward_func(
        completions=[
            [{"content": "This is my answer"}],
            [{"content": "<reasoning>\nThis is my reasoning.\n</reasoning>\n<answer>\n3\n</answer>"}],
        ]
    )
    assert res[1] > res[0], f"format: {res}"
    print(f"PASS format_reward_func {res}")


def assert_correct_answer_reward() -> None:
    """A wrong final answer must score below a correct one."""
    res = correct_answer_reward_func(
        prompts=[
            [{"content": "How many..."}],
            [{"content": "How many..."}],
        ],
        completions=[
            [{"content": "<reasoning>.../reasoning>\n<answer>\n3\n</answer>"}],
            [{"content": "<reasoning>.../reasoning>\n<answer>\n3\n</answer>"}],
        ],
        counts=[0, 3],
    )
    assert res[1] > res[0], f"correct_answer: {res}"
    print(f"PASS correct_answer_reward_func {res}")


REWARD_ASSERTS = [
    assert_numbering_reward,
    assert_spelling_reward,
    assert_counting_reward,
    assert_format_reward,
    assert_correct_answer_reward,
]


def run_all_reward_asserts() -> None:
    """Run the extractor checks and every per-reward good-vs-bad assertion.

    This is the rubric-evidence entry point: notebook cells call the individual
    assert_* helpers; `python src/rewards.py` and smoke_test.py call this.
    """
    assert_extractors()
    for fn in REWARD_ASSERTS:
        fn()


if __name__ == "__main__":
    run_all_reward_asserts()
    print("\nALL REWARD ASSERTS PASSED")
