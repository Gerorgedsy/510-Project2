"""Download the GSM8K test set and save it to data/raw/gsm8k_test.csv."""

from pathlib import Path

import pandas as pd
from datasets import load_dataset

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "gsm8k_test.csv"


def extract_reference(answer_text: str) -> str:
    """GSM8K solutions end with '#### <number>'; keep just the number."""
    return answer_text.split("####")[-1].strip().replace(",", "")


def main() -> None:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    df = pd.DataFrame(
        {
            "question_id": range(len(ds)),
            "question": ds["question"],
            "reference_solution": ds["answer"],
        }
    )
    df["reference_answer"] = df["reference_solution"].map(extract_reference)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} questions to {OUT_PATH}")


if __name__ == "__main__":
    main()
