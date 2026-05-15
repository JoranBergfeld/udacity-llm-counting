"""
No-GPU sanity check: rewards asserts + dataset record generation.

Usage: python src/smoke_test.py

Exits non-zero if anything fails. Should complete in well under a second.
"""

import sys

from dataset import ALL_WORDS, generate_records
from rewards import run_all_reward_asserts


def main() -> None:
    run_all_reward_asserts()

    records = list(generate_records())
    assert len(records) > 0, "generate_records produced nothing"
    assert all("words" in r and "letters" in r and "counts" in r for r in records), (
        "malformed records"
    )
    in_word = [r for r in records if r["counts"] > 0]
    out_of_word = [r for r in records if r["counts"] == 0]
    assert in_word and out_of_word, "expected both in-word and out-of-word records"
    print(f"\nPASS dataset: {len(records)} records from {len(ALL_WORDS)} words "
          f"({len(in_word)} in-word, {len(out_of_word)} out-of-word)")

    print("\nALL OK")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"FAIL: {e}")
        sys.exit(1)
