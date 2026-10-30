"""Statistical analysis: McNemar's test, effect sizes and confidence intervals.

Reads data/processed/paired.csv and writes results/summary.json and figures.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.contingency_tables import mcnemar

ROOT = Path(__file__).resolve().parents[1]
PAIRED_PATH = ROOT / "data" / "processed" / "paired.csv"
RESULTS = ROOT / "results"


def paired_diff_ci(b: int, c: int, n: int, alpha: float = 0.05):
    """Wald CI for the difference in paired proportions (tip - no tip)."""
    diff = (b - c) / n
    se = np.sqrt(b + c - (b - c) ** 2 / n) / n
    z = norm.ppf(1 - alpha / 2)
    return diff, diff - z * se, diff + z * se


def main() -> None:
    df = pd.read_csv(PAIRED_PATH)
    n = len(df)
    tip, no_tip = df["tip_100"], df["tip_0"]

    a = int(((tip == 1) & (no_tip == 1)).sum())  # both correct
    b = int(((tip == 1) & (no_tip == 0)).sum())  # correct only with tip
    c = int(((tip == 0) & (no_tip == 1)).sum())  # correct only without tip
    d = int(((tip == 0) & (no_tip == 0)).sum())  # both wrong
    table = [[a, b], [c, d]]

    exact = (b + c) < 25
    test = mcnemar(table, exact=exact, correction=not exact)
    diff, lo, hi = paired_diff_ci(b, c, n)

    summary = {
        "n_pairs": n,
        "accuracy_tip_0": float(no_tip.mean()),
        "accuracy_tip_100": float(tip.mean()),
        "contingency_table": {"both_correct": a, "only_tip_100": b, "only_tip_0": c, "both_wrong": d},
        "mcnemar_exact": exact,
        "mcnemar_statistic": float(test.statistic),
        "p_value": float(test.pvalue),
        "accuracy_difference": diff,
        "difference_95ci": [lo, hi],
        "paired_odds_ratio_b_over_c": (b / c) if c else None,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    # Figure: accuracy by condition
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["$0 tip", "$100 tip"], [no_tip.mean(), tip.mean()], color=["#7a8b99", "#2f6db5"])
    ax.set_ylabel("Accuracy on GSM8K test set")
    ax.set_ylim(0, 1)
    ax.set_title(f"Accuracy by prompt condition (n = {n})")
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "accuracy_by_condition.png", dpi=200)


if __name__ == "__main__":
    main()
