"""Query the model for every (question, condition) pair of one stage.

Raw outputs are appended to data/raw/<stage>_outputs.jsonl, one line per pair.
Pairs already in the file are skipped, so the script can be stopped and rerun safely;
a failed request is logged and picked up by the next run.

Usage: python src/collect.py pilot|main
"""

import argparse
import asyncio
import json
import random
from datetime import datetime, timezone

import pandas as pd

from config import (
    CONCURRENCY,
    ENABLE_THINKING,
    MAX_TOKENS,
    MODEL_NAME,
    ORDER_SEED,
    STAGES,
    TEMPERATURE,
    outputs_path,
    questions_path,
)
from prompts import CONDITIONS, build_prompt
from sampler import generate, make_sampler


def load_done(path) -> set:
    if not path.exists():
        return set()
    with path.open() as f:
        return {(r["question_id"], r["condition"]) for r in map(json.loads, f)}


async def collect(stage: str) -> None:
    questions = pd.read_csv(questions_path(stage)).set_index("question_id")
    out_path = outputs_path(stage)

    jobs = [(qid, cond) for qid in questions.index for cond in CONDITIONS]
    random.Random(ORDER_SEED).shuffle(jobs)  # interleave questions and conditions in random order
    done = load_done(out_path)
    jobs = [j for j in jobs if j not in done]
    print(f"[{stage}] {len(done)} already done, {len(jobs)} remaining")
    if not jobs:
        return

    client, tokenizer = await make_sampler()
    sem = asyncio.Semaphore(CONCURRENCY)
    n_written = n_failed = 0

    async def one(qid, cond, f) -> None:
        nonlocal n_written, n_failed
        prompt = build_prompt(questions.loc[qid], cond)
        async with sem:
            try:
                out = await generate(client, tokenizer, prompt)
            except Exception as e:  # keep going; rerunning the script retries it
                n_failed += 1
                print(f"question {qid} / {cond} failed: {e!r}")
                return
        record = {
            "question_id": int(qid),
            "condition": cond,
            "model": MODEL_NAME,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "enable_thinking": ENABLE_THINKING,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "prompt": prompt,
            **out,
        }
        f.write(json.dumps(record) + "\n")
        f.flush()
        n_written += 1
        if n_written % 100 == 0:
            print(f"{n_written}/{len(jobs)} done")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        await asyncio.gather(*(one(qid, cond, f) for qid, cond in jobs))
    print(f"[{stage}] finished: {n_written} written, {n_failed} failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=STAGES)
    asyncio.run(collect(parser.parse_args().stage))


if __name__ == "__main__":
    main()
