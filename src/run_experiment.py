"""Query the model on every GSM8K test question under both conditions.

Raw outputs are appended to data/raw/model_outputs.jsonl, one line per
(question, condition). Already-finished pairs are skipped, so the script
can be stopped and restarted safely.
"""

import json
import os
import random
from pathlib import Path

import pandas as pd

from prompts import CONDITIONS, build_prompt

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "data" / "raw" / "gsm8k_test.csv"
OUTPUT_PATH = ROOT / "data" / "raw" / "model_outputs.jsonl"

MODEL_NAME = "TODO: fill in the exact Tinker model name"
TEMPERATURE = 0.0  # greedy decoding -> deterministic, reproducible outputs
MAX_TOKENS = 512
SEED = 510


def query_model(prompt: str) -> str:
    """Send one prompt to the model via Tinker and return its text response.

    TODO: implement with the Tinker sampling client, using MODEL_NAME,
    TEMPERATURE and MAX_TOKENS. Requires the TINKER_API_KEY env variable.
    """
    raise NotImplementedError("Tinker call not implemented yet")


def load_done() -> set:
    if not OUTPUT_PATH.exists():
        return set()
    with OUTPUT_PATH.open() as f:
        return {(r["question_id"], r["condition"]) for r in map(json.loads, f)}


def main() -> None:
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("Please set the TINKER_API_KEY environment variable.")

    questions = pd.read_csv(QUESTIONS_PATH)
    jobs = [(qid, cond) for qid in questions["question_id"] for cond in CONDITIONS]
    random.Random(SEED).shuffle(jobs)  # randomize question and condition order

    done = load_done()
    jobs = [j for j in jobs if j not in done]
    print(f"{len(done)} already done, {len(jobs)} remaining")

    q_lookup = questions.set_index("question_id")["question"]
    with OUTPUT_PATH.open("a") as f:
        for i, (qid, cond) in enumerate(jobs, 1):
            prompt = build_prompt(q_lookup[qid], cond)
            response = query_model(prompt)
            record = {
                "question_id": int(qid),
                "condition": cond,
                "model": MODEL_NAME,
                "temperature": TEMPERATURE,
                "prompt": prompt,
                "response": response,
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            if i % 50 == 0:
                print(f"{i}/{len(jobs)} done")


if __name__ == "__main__":
    main()
