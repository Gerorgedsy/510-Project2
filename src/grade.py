"""Extract each model answer and score it against the GSM8K reference.

Grading rule (fixed before data collection):
  * Take the number after the LAST 'Answer:' in the response.
  * Remove commas, '$' and trailing periods; compare numerically.
  * If no 'Answer:' line or no number can be parsed -> incorrect.

Output: data/processed/graded_long.csv  (one row per question x condition)
        data/processed/paired.csv       (one row per question, both conditions)
"""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "data" / "raw" / "gsm8k_test.csv"
OUTPUTS_PATH = ROOT / "data" / "raw" / "model_outputs.jsonl"
PROCESSED = ROOT / "data" / "processed"

ANSWER_RE = re.compile(r"Answer:\s*\$?\s*(-?[\d,]*\.?\d+)", re.IGNORECASE)


def extract_answer(response: str):
    """Return the parsed final answer as a float, or None if unparseable."""
    matches = ANSWER_RE.findall(response or "")
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def is_correct(pred, reference: str) -> bool:
    if pred is None:
        return False
    return abs(pred - float(reference)) < 1e-6


def main() -> None:
    questions = pd.read_csv(QUESTIONS_PATH, dtype={"reference_answer": str})
    with OUTPUTS_PATH.open() as f:
        outputs = pd.DataFrame([json.loads(line) for line in f])

    df = outputs.merge(questions[["question_id", "reference_answer"]], on="question_id")
    df["predicted"] = df["response"].map(extract_answer)
    df["parse_failed"] = df["predicted"].isna()
    df["correct"] = [is_correct(p, r) for p, r in zip(df["predicted"], df["reference_answer"])]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["prompt"]).to_csv(PROCESSED / "graded_long.csv", index=False)

    paired = df.pivot(index="question_id", columns="condition", values="correct")
    paired = paired.dropna().astype(int).reset_index()
    paired.to_csv(PROCESSED / "paired.csv", index=False)

    print(f"Graded {len(df)} responses; {len(paired)} complete pairs")
    print(f"Parse failures: {df['parse_failed'].sum()}")
    print(df.groupby("condition")["correct"].mean().rename("accuracy"))


if __name__ == "__main__":
    main()
