"""Score each answer and compute length measures for one stage.

Grading rule (fixed on pilot data, before the main run):
  * Take the number after the LAST 'Answer:' label. Markdown bold, '$' and LaTeX
    '\\boxed{...}' around the label or the number are allowed.
  * If there is no 'Answer:' label, fall back to the LAST '\\boxed{<number>}'.
  * Remove commas; compare numerically with the GSM8K reference.
  * Unparseable or truncated answers count as incorrect.

Outputs:
  data/processed/<stage>_long.csv   one row per (question, condition)
  data/processed/<stage>_wide.csv   one row per question: tokens_<cond>, correct_<cond>, truncated_<cond>

Usage: python src/grade.py pilot|main
"""

import argparse
import json
import re

import pandas as pd

from config import STAGES, long_path, outputs_path, questions_path, wide_path

# 'Answer:' / '**Answer:**' / '**Answer**:', then optional '**', '$', '\boxed{' before the number
ANSWER_RE = re.compile(r"Answer\s*\**\s*:[\s*$]*(?:\\boxed\{)?[\s*$]*(-?[\d,]*\.?\d+)", re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{\s*\$?\s*(-?[\d,]*\.?\d+)")
TIP_MENTION_RE = re.compile(r"\btip(?:s|ping|ped)?\b", re.IGNORECASE)


def extract_answer(response: str):
    """Return the parsed final answer as a float, or None if unparseable."""
    matches = ANSWER_RE.findall(response or "") or BOXED_RE.findall(response or "")
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", ""))
    except ValueError:
        return None


def is_correct(pred, reference: str) -> bool:
    if pred is None or pd.isna(pred):
        return False
    return abs(pred - float(reference)) < 1e-6


def grade(stage: str) -> None:
    questions = pd.read_csv(questions_path(stage), dtype={"reference_answer": str})
    with outputs_path(stage).open() as f:
        outputs = pd.DataFrame([json.loads(line) for line in f])

    df = outputs.merge(questions[["question_id", "reference_answer"]], on="question_id")
    df["predicted"] = df["response"].map(extract_answer)
    df["parse_failed"] = df["predicted"].isna()
    df["truncated"] = df["stop_reason"] == "length"
    df["correct"] = [is_correct(p, r) and not t for p, r, t in zip(df["predicted"], df["reference_answer"], df["truncated"])]
    df["tokens"] = df["n_output_tokens"]
    df["n_chars"] = df["response"].str.len()
    df["n_words"] = df["response"].str.split().str.len()
    df["mentions_tip"] = df["response"].str.contains(TIP_MENTION_RE)

    long_path(stage).parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["prompt", "response"]).to_csv(long_path(stage), index=False)

    wide = df.pivot(index="question_id", columns="condition", values=["tokens", "correct", "truncated"])
    wide.columns = [f"{value}_{cond}" for value, cond in wide.columns]
    wide = wide.dropna().reset_index()  # keep questions answered under every condition
    for col in wide.columns:
        if col.startswith("tokens_"):
            wide[col] = wide[col].astype(int)
        elif col.startswith(("correct_", "truncated_")):
            wide[col] = wide[col].astype(bool)
    wide.to_csv(wide_path(stage), index=False)

    print(f"[{stage}] graded {len(df)} answers; {len(wide)} questions complete under all conditions")
    print(f"parse failures: {int(df['parse_failed'].sum())}; truncated at max_tokens: {int(df['truncated'].sum())}")
    print(df.groupby("condition").agg(accuracy=("correct", "mean"), mean_tokens=("tokens", "mean")).round(3))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=STAGES)
    grade(parser.parse_args().stage)


if __name__ == "__main__":
    main()
