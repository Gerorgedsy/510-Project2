"""Build the paired question sets (original vs. distractor) and save them to data/raw/.

  * original question + reference answer: GSM8K test split
  * distractor question: GSM-Plus "distraction insertion" variant of the same problem,
    matched on the exact seed-question text. GSM-Plus keeps the answer unchanged for this
    perturbation, so both versions are scored against the GSM8K reference.
  * distractor_numbers: numbers written in digits that appear only in the distractor version

Usage: python src/load_data.py
"""

import random
import re

import pandas as pd
from datasets import load_dataset

from config import N_PILOT, PILOT_SEED, questions_path

NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> set:
    return {n.replace(",", "") for n in NUMBER_RE.findall(text)}


def reference_answer(solution: str) -> str:
    """GSM8K solutions end with '#### <number>'; keep just the number."""
    return solution.split("####")[-1].strip().replace(",", "")


def main() -> None:
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    plus = load_dataset("qintongli/GSM-Plus", split="test")
    distract = {r["seed_question"].strip(): r for r in plus if r["perturbation_type"] == "distraction insertion"}

    rows = []
    for qid, (question, solution) in enumerate(zip(gsm8k["question"], gsm8k["answer"])):
        d = distract[question.strip()]
        ref = reference_answer(solution)
        if float(str(d["answer"]).replace(",", "")) != float(ref):
            raise ValueError(f"GSM-Plus answer differs from GSM8K for question {qid}")
        extra = numbers_in(d["question"]) - numbers_in(question)
        rows.append(
            {
                "question_id": qid,  # index in the GSM8K test split
                "original_question": question,
                "distractor_question": d["question"],
                "reference_answer": ref,
                "distractor_numbers": ";".join(sorted(extra)),
            }
        )
    df = pd.DataFrame(rows)

    pilot_ids = set(random.Random(PILOT_SEED).sample(list(df["question_id"]), N_PILOT))
    frames = {"pilot": df[df["question_id"].isin(pilot_ids)], "main": df[~df["question_id"].isin(pilot_ids)]}
    for stage, part in frames.items():
        path = questions_path(stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        part.to_csv(path, index=False)
        print(f"{stage}: saved {len(part)} question pairs -> {path}")
    print(f"distractor adds at least one new digit-number in {int((df['distractor_numbers'] != '').sum())} of {len(df)} problems")


if __name__ == "__main__":
    main()
