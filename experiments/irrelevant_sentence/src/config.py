"""Settings and file paths shared by every script in this experiment folder.

The only file read from outside this folder is the repo-root .env (TINKER_API_KEY).
Questions are the 1,319 GSM8K test problems paired with their GSM-Plus "distraction insertion"
versions. A random N_PILOT of them form the pilot; the rest form the main study (no overlap).
"""

from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
REPO_ENV = EXP_DIR.parents[1] / ".env"

DATA_RAW = EXP_DIR / "data" / "raw"
DATA_PROCESSED = EXP_DIR / "data" / "processed"
RESULTS = EXP_DIR / "results"

STAGES = ("pilot", "main")

# model + decoding, identical for every condition
MODEL_NAME = "Qwen/Qwen3.5-4B"
ENABLE_THINKING = False  # Qwen3.5 hybrid thinking off -> plain step-by-step answer
TEMPERATURE = 0.0  # greedy decoding (Tinker is still not bit-for-bit deterministic)
MAX_TOKENS = 2048  # generous cap so long answers are not cut off
SAMPLING_SEED = 510
CONCURRENCY = 32  # requests in flight at once

ORDER_SEED = 510  # shuffles the (question, condition) call order
PILOT_SEED = 510  # picks the pilot questions
N_PILOT = 100


def questions_path(stage: str) -> Path:
    return DATA_RAW / f"{stage}_questions.csv"


def outputs_path(stage: str) -> Path:
    return DATA_RAW / f"{stage}_outputs.jsonl"


def long_path(stage: str) -> Path:
    return DATA_PROCESSED / f"{stage}_long.csv"


def wide_path(stage: str) -> Path:
    return DATA_PROCESSED / f"{stage}_wide.csv"


def results_dir(stage: str) -> Path:
    return RESULTS / stage
