"""Statistical analysis for one stage.

Primary (pre-registered): accuracy, distractor vs original.
  McNemar's test (two-sided, alpha = 0.05; exact binomial if < 25 discordant pairs),
  paired accuracy difference with 95% CI, conditional odds ratio with exact 95% CI.
  Sensitivity: the same test without questions where either answer was truncated.
Secondary: answer length, d = log(tokens_distractor) - log(tokens_original); paired t-test,
  Wilcoxon signed-rank as robustness.
Exploratory: among problems whose inserted sentence adds a new number, how often the distractor
  answer uses that number (vs. the coincidence rate in the original answer), and whether using
  it goes with wrong answers (Fisher's exact test).
Negative control (A/A): original_repeat vs original, the same accuracy and length tests.

Usage: python src/analyze.py pilot|main
Writes results/<stage>/summary.json and figures to results/<stage>/.
"""

import argparse
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.contingency_tables import mcnemar  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402

from config import STAGES, long_path, results_dir, wide_path  # noqa: E402

ALPHA = 0.05
CONDITIONS = ["original", "distractor", "original_repeat"]


def accuracy_test(wide: pd.DataFrame, a: str, b: str, drop_truncated: bool = False) -> dict:
    """Paired comparison of correctness, condition a vs condition b (difference = a - b)."""
    sub = wide
    if drop_truncated:
        sub = wide[~(wide[f"truncated_{a}"] | wide[f"truncated_{b}"])]
    x, y = sub[f"correct_{a}"].astype(bool), sub[f"correct_{b}"].astype(bool)
    n = len(sub)
    both, only_a, only_b, neither = int((x & y).sum()), int((x & ~y).sum()), int((~x & y).sum()), int((~x & ~y).sum())
    disc = only_a + only_b
    exact = disc < 25
    test = mcnemar([[both, only_a], [only_b, neither]], exact=exact, correction=not exact)
    diff = (only_a - only_b) / n
    se = math.sqrt(disc - (only_a - only_b) ** 2 / n) / n
    z = stats.norm.ppf(1 - ALPHA / 2)
    if disc:  # conditional odds ratio only_a / only_b, exact CI from the binomial share only_a / disc
        p_lo, p_hi = proportion_confint(only_a, disc, alpha=ALPHA, method="beta")
        odds = lambda p: p / (1 - p) if p < 1 else math.inf  # noqa: E731
        or_ci = [odds(p_lo), odds(p_hi)]
    else:
        or_ci = [None, None]
    return {
        "comparison": f"{a} vs {b}",
        "n": n,
        f"accuracy_{a}": float(x.mean()),
        f"accuracy_{b}": float(y.mean()),
        "table": {"both_correct": both, f"only_{a}": only_a, f"only_{b}": only_b, "both_wrong": neither},
        "discordant_share": disc / n,
        "mcnemar_exact": exact,
        "mcnemar_statistic": float(test.statistic),
        "p_mcnemar": float(test.pvalue),
        "accuracy_difference": diff,
        "difference_95ci": [diff - z * se, diff + z * se],
        "conditional_odds_ratio": (only_a / only_b) if only_b else None,
        "odds_ratio_95ci": or_ci,
    }


def length_test(wide: pd.DataFrame, a: str, b: str) -> dict:
    """Paired comparison of log answer length, condition a vs condition b."""
    d = np.log(wide[f"tokens_{a}"].to_numpy()) - np.log(wide[f"tokens_{b}"].to_numpy())
    t = stats.ttest_1samp(d, 0.0)
    lo, hi = t.confidence_interval(1 - ALPHA)
    w = stats.wilcoxon(d) if np.any(d != 0) else None
    return {
        "comparison": f"{a} vs {b}",
        "n": int(len(d)),
        "length_ratio": math.exp(d.mean()),
        "length_ratio_95ci": [math.exp(lo), math.exp(hi)],
        "cohens_dz": float(d.mean() / d.std(ddof=1)),
        "t": float(t.statistic),
        "df": int(t.df),
        "p_ttest": float(t.pvalue),
        "p_wilcoxon": float(w.pvalue) if w else None,
        "share_longer": float((d > 0).mean()),
        "share_shorter": float((d < 0).mean()),
    }


def distractor_use(wide: pd.DataFrame) -> dict:
    """Exploratory: does using the distractor's number go with wrong answers?"""
    sub = wide[wide["has_distractor_number"].astype(bool)].copy()
    if sub.empty:
        return {"n_with_new_number": 0}
    uses = sub["uses_number_distractor"].astype(bool)
    wrong = ~sub["correct_distractor"].astype(bool)
    table = [[int((uses & wrong).sum()), int((uses & ~wrong).sum())], [int((~uses & wrong).sum()), int((~uses & ~wrong).sum())]]
    odds, p = stats.fisher_exact(table)
    flipped = sub["correct_original"].astype(bool) & wrong  # right without the distractor, wrong with it
    return {
        "n_with_new_number": int(len(sub)),
        "share_using_number": float(uses.mean()),
        "share_number_in_original_answer": float(sub["uses_number_original"].astype(bool).mean()),
        "table_uses_x_wrong": {"uses_wrong": table[0][0], "uses_right": table[0][1], "ignores_wrong": table[1][0], "ignores_right": table[1][1]},
        "fisher_odds_ratio": float(odds),
        "p_fisher": float(p),
        "flipped_to_wrong": int(flipped.sum()),
        "flipped_and_used_number": int((flipped & uses).sum()),
    }


def make_figures(long: pd.DataFrame, wide: pd.DataFrame, out_dir) -> None:
    # 1. discordant pairs: distractor vs original, against A/A noise
    pairs = [("distractor", "original"), ("original_repeat", "original")]
    labels, worse, better = [], [], []
    for a, b in pairs:
        x, y = wide[f"correct_{a}"].astype(bool), wide[f"correct_{b}"].astype(bool)
        labels.append(f"{a}\nvs {b}")
        worse.append(int((~x & y).sum()))
        better.append(int((x & ~y).sum()))
    pos = np.arange(len(pairs))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(pos - 0.18, worse, width=0.36, color="#c0504d", label="right -> wrong")
    ax.bar(pos + 0.18, better, width=0.36, color="#4f81bd", label="wrong -> right")
    ax.set_xticks(pos, labels)
    ax.set_ylabel("questions")
    ax.set_title(f"Questions whose correctness changed (n = {len(wide)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "discordant_pairs.png", dpi=200)
    plt.close(fig)

    # 2. accuracy per condition with Wilson 95% CI (unpaired CIs: read the paired test, not the overlap)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for i, cond in enumerate(CONDITIONS):
        s = long.loc[long["condition"] == cond, "correct"]
        k, n = int(s.sum()), len(s)
        lo, hi = proportion_confint(k, n, method="wilson")
        ax.errorbar(i, k / n, yerr=[[k / n - lo], [hi - k / n]], fmt="o", capsize=4, color="#c0504d" if cond == "distractor" else "#555555")
    ax.set_xticks(range(len(CONDITIONS)), CONDITIONS)
    ax.set_ylabel("accuracy (Wilson 95% CI)")
    ax.set_title("Accuracy by condition")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_by_condition.png", dpi=200)
    plt.close(fig)


def analyze(stage: str) -> None:
    long = pd.read_csv(long_path(stage))
    wide = pd.read_csv(wide_path(stage))
    out_dir = results_dir(stage)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "stage": stage,
        "n_questions": len(wide),
        "descriptives": long.groupby("condition")
        .agg(n=("correct", "size"), accuracy=("correct", "mean"), mean_tokens=("tokens", "mean"), truncated=("truncated", "sum"), parse_failed=("parse_failed", "sum"))
        .reindex(CONDITIONS)
        .reset_index()
        .to_dict(orient="records"),
        "primary_accuracy_distractor_vs_original": accuracy_test(wide, "distractor", "original"),
        "sensitivity_accuracy_without_truncated": accuracy_test(wide, "distractor", "original", drop_truncated=True),
        "secondary_length_distractor_vs_original": length_test(wide, "distractor", "original"),
        "exploratory_distractor_number_use": distractor_use(wide),
        "negative_control_accuracy_aa": accuracy_test(wide, "original_repeat", "original"),
        "negative_control_length_aa": length_test(wide, "original_repeat", "original"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    make_figures(long, wide, out_dir)

    print(f"[{stage}] n = {len(wide)} questions\n")
    print(pd.DataFrame(summary["descriptives"]).round(3).to_string(index=False), "\n")
    for name, key in [("PRIMARY accuracy distractor vs original", "primary_accuracy_distractor_vs_original"), ("  without truncated", "sensitivity_accuracy_without_truncated"), ("A/A accuracy original_repeat vs original", "negative_control_accuracy_aa")]:
        r = summary[key]
        lo, hi = r["difference_95ci"]
        or_ = f"{r['conditional_odds_ratio']:.2f}" if r["conditional_odds_ratio"] is not None else "n/a"
        print(f"{name:42s} diff {r['accuracy_difference']:+.3f} [{lo:+.3f}, {hi:+.3f}]  table {r['table']}  McNemar p = {r['p_mcnemar']:.4g}  OR {or_}")
    for name, key in [("length distractor vs original", "secondary_length_distractor_vs_original"), ("A/A length", "negative_control_length_aa")]:
        r = summary[key]
        lo, hi = r["length_ratio_95ci"]
        pw = f"{r['p_wilcoxon']:.4g}" if r["p_wilcoxon"] is not None else "n/a"
        print(f"{name:42s} ratio {r['length_ratio']:.3f} [{lo:.3f}, {hi:.3f}]  t p = {r['p_ttest']:.4g}  wilcoxon p = {pw}")
    print("\nexploratory distractor-number use:", json.dumps(summary["exploratory_distractor_number_use"]))
    print(f"\nWrote {out_dir / 'summary.json'} and figures")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=STAGES)
    analyze(parser.parse_args().stage)


if __name__ == "__main__":
    main()
