"""Download GSM8K and save the question sets for both stages to data/raw/.

  * pilot: N_PILOT random questions from the train split (never overlaps the test set)
  * main:  all 1,319 questions of the test split

Usage: python src/load_data.py
"""

import random

import pandas as pd
from datasets import load_dataset

from config import N_PILOT, PILOT_SEED, questions_path


def reference_answer(solution: str) -> str:
    """GSM8K solutions end with '#### <number>'; keep just the number."""
    return solution.split("####")[-1].strip().replace(",", "")


def to_frame(split, ids) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question_id": ids,  # index in the original GSM8K split
            "question": [split[i]["question"] for i in ids],
            "reference_answer": [reference_answer(split[i]["answer"]) for i in ids],
        }
    )


def main() -> None:
    train = load_dataset("openai/gsm8k", "main", split="train")
    test = load_dataset("openai/gsm8k", "main", split="test")

    pilot_ids = sorted(random.Random(PILOT_SEED).sample(range(len(train)), N_PILOT))
    frames = {"pilot": to_frame(train, pilot_ids), "main": to_frame(test, list(range(len(test))))}

    for stage, df in frames.items():
        path = questions_path(stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"{stage}: saved {len(df)} questions -> {path}")


if __name__ == "__main__":
    main()
