"""Statistical analysis for one stage.

Primary (pre-registered): answer length, tip_100 vs tip_0.
  d = log(tokens_tip_100) - log(tokens_tip_0) per question; two-sided paired t-test at
  alpha = 0.05; geometric-mean length ratio with 95% CI; Cohen's dz.
  Robustness: Wilcoxon signed-rank test with matched-pairs rank-biserial r.
  Sensitivity: the same tests without questions where either answer was truncated.
Secondary:
  * length across none / tip_0 / tip_100: Friedman test, then pairwise Wilcoxon (Holm)
  * accuracy tip_100 vs tip_0: McNemar (exact if < 25 discordant pairs), paired difference
    with 95% CI, TOST equivalence test with a +-2 point margin
Negative control (A/A): tip_0 vs tip_0_repeat, the same length and accuracy tests;
any difference there is run-to-run noise from the server.

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
from statsmodels.stats.multitest import multipletests  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402

from config import STAGES, long_path, results_dir, wide_path  # noqa: E402

ALPHA = 0.05
EQUIV_MARGIN = 0.02  # accuracy differences within +-2 points count as practically equivalent
CONDITIONS = ["none", "tip_0", "tip_100", "tip_0_repeat"]


def length_test(wide: pd.DataFrame, a: str, b: str, drop_truncated: bool = False) -> dict:
    """Paired comparison of log answer length, condition a vs condition b."""
    sub = wide
    if drop_truncated:
        sub = wide[~(wide[f"truncated_{a}"] | wide[f"truncated_{b}"])]
    d = np.log(sub[f"tokens_{a}"].to_numpy()) - np.log(sub[f"tokens_{b}"].to_numpy())
    t = stats.ttest_1samp(d, 0.0)
    lo, hi = t.confidence_interval(1 - ALPHA)
    w = stats.wilcoxon(d)
    nz = d[d != 0]
    ranks = stats.rankdata(np.abs(nz))
    return {
        "comparison": f"{a} vs {b}",
        "n": int(len(d)),
        "mean_log_ratio": float(d.mean()),
        "sd_log_ratio": float(d.std(ddof=1)),
        "length_ratio": math.exp(d.mean()),
        "length_ratio_95ci": [math.exp(lo), math.exp(hi)],
        "cohens_dz": float(d.mean() / d.std(ddof=1)),
        "t": float(t.statistic),
        "df": int(t.df),
        "p_ttest": float(t.pvalue),
        "wilcoxon_W": float(w.statistic),
        "p_wilcoxon": float(w.pvalue),
        "rank_biserial_r": float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()),
        "share_longer": float((d > 0).mean()),
        "share_shorter": float((d < 0).mean()),
        "shapiro_W": float(stats.shapiro(d).statistic),
        "shapiro_p": float(stats.shapiro(d).pvalue),
        "skewness": float(stats.skew(d)),
    }


def accuracy_test(wide: pd.DataFrame, a: str, b: str) -> dict:
    """Paired comparison of correctness, condition a vs condition b."""
    x, y = wide[f"correct_{a}"].astype(bool), wide[f"correct_{b}"].astype(bool)
    n = len(x)
    both, only_a, only_b, neither = int((x & y).sum()), int((x & ~y).sum()), int((~x & y).sum()), int((~x & ~y).sum())
    disc = only_a + only_b
    exact = disc < 25
    test = mcnemar([[both, only_a], [only_b, neither]], exact=exact, correction=not exact)
    diff = (only_a - only_b) / n
    se = math.sqrt(disc - (only_a - only_b) ** 2 / n) / n
    z95, z90 = stats.norm.ppf(0.975), stats.norm.ppf(0.95)
    if se > 0:
        p_tost = max(1 - stats.norm.cdf((diff + EQUIV_MARGIN) / se), stats.norm.cdf((diff - EQUIV_MARGIN) / se))
    else:  # no discordant pairs: the difference is exactly 0
        p_tost = 0.0
    ci90 = [diff - z90 * se, diff + z90 * se]
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
        "difference_95ci": [diff - z95 * se, diff + z95 * se],
        "difference_90ci": ci90,
        "paired_odds_ratio": (only_a / only_b) if only_b else None,
        "tost_margin": EQUIV_MARGIN,
        "p_tost": float(p_tost),
        "equivalent_within_margin": bool(-EQUIV_MARGIN < ci90[0] and ci90[1] < EQUIV_MARGIN),
    }


def descriptives(long: pd.DataFrame) -> pd.DataFrame:
    g = long.groupby("condition")
    out = g.agg(
        n=("tokens", "size"),
        mean_tokens=("tokens", "mean"),
        median_tokens=("tokens", "median"),
        geo_mean_tokens=("tokens", lambda s: float(np.exp(np.log(s).mean()))),
        accuracy=("correct", "mean"),
        parse_failed=("parse_failed", "sum"),
        truncated=("truncated", "sum"),
        mentions_tip=("mentions_tip", "mean"),
    )
    return out.reindex(CONDITIONS)


def make_figures(long: pd.DataFrame, wide: pd.DataFrame, out_dir) -> None:
    # 1. per-question log length ratio: tip effect vs A/A noise
    d_tip = np.log(wide["tokens_tip_100"]) - np.log(wide["tokens_tip_0"])
    d_aa = np.log(wide["tokens_tip_0_repeat"]) - np.log(wide["tokens_tip_0"])
    lim = max(abs(d_tip).quantile(0.99), abs(d_aa).quantile(0.99))
    bins = np.linspace(-lim, lim, 41)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(d_aa, bins=bins, alpha=0.6, color="#9aa5b1", label="A/A noise: tip_0 repeat vs tip_0")
    ax.hist(d_tip, bins=bins, histtype="step", linewidth=2, color="#2f6db5", label="tip_100 vs tip_0")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(d_tip.mean(), color="#2f6db5", linestyle="--", linewidth=1.2, label=f"mean = {d_tip.mean():+.3f}")
    ax.set_xlabel("log(answer tokens, condition / tip_0)")
    ax.set_ylabel("questions")
    ax.set_title(f"Per-question change in answer length (n = {len(wide)})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "length_ratio_hist.png", dpi=200)
    plt.close(fig)

    # 2. geometric-mean answer length per condition with 95% CI
    rows = []
    for cond in CONDITIONS:
        logs = np.log(long.loc[long["condition"] == cond, "tokens"])
        half = stats.t.ppf(0.975, len(logs) - 1) * logs.std(ddof=1) / math.sqrt(len(logs))
        rows.append((cond, math.exp(logs.mean()), math.exp(logs.mean() - half), math.exp(logs.mean() + half)))
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for i, (cond, mid, lo, hi) in enumerate(rows):
        ax.errorbar(i, mid, yerr=[[mid - lo], [hi - mid]], fmt="o", capsize=4, color="#2f6db5" if cond == "tip_100" else "#555555")
    ax.set_xticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylabel("geometric-mean answer tokens (95% CI)")
    ax.set_title("Answer length by condition")
    fig.tight_layout()
    fig.savefig(out_dir / "tokens_by_condition.png", dpi=200)
    plt.close(fig)

    # 3. accuracy per condition with Wilson 95% CI
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for i, cond in enumerate(CONDITIONS):
        s = long.loc[long["condition"] == cond, "correct"]
        k, n = int(s.sum()), len(s)
        lo, hi = proportion_confint(k, n, method="wilson")
        ax.errorbar(i, k / n, yerr=[[k / n - lo], [hi - k / n]], fmt="o", capsize=4, color="#2f6db5" if cond == "tip_100" else "#555555")
    ax.set_xticks(range(len(CONDITIONS)), CONDITIONS)
    ax.set_ylabel("accuracy (Wilson 95% CI)")
    ax.set_title("GSM8K accuracy by condition")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_by_condition.png", dpi=200)
    plt.close(fig)


def analyze(stage: str) -> None:
    long = pd.read_csv(long_path(stage))
    wide = pd.read_csv(wide_path(stage))
    out_dir = results_dir(stage)
    out_dir.mkdir(parents=True, exist_ok=True)

    primary = length_test(wide, "tip_100", "tip_0")
    primary_no_trunc = length_test(wide, "tip_100", "tip_0", drop_truncated=True)

    logs = [np.log(wide[f"tokens_{c}"]) for c in ("none", "tip_0", "tip_100")]
    fr = stats.friedmanchisquare(*logs)
    pairs = [("tip_100", "tip_0"), ("tip_100", "none"), ("tip_0", "none")]
    posthoc = [length_test(wide, a, b) for a, b in pairs]
    holm = multipletests([p["p_wilcoxon"] for p in posthoc], method="holm")[1]
    for p, adj in zip(posthoc, holm):
        p["p_wilcoxon_holm"] = float(adj)

    summary = {
        "stage": stage,
        "n_questions": len(wide),
        "descriptives": descriptives(long).reset_index().to_dict(orient="records"),
        "primary_length_tip100_vs_tip0": primary,
        "sensitivity_length_without_truncated": primary_no_trunc,
        "secondary_length_friedman": {"conditions": ["none", "tip_0", "tip_100"], "chi2": float(fr.statistic), "p": float(fr.pvalue), "posthoc_wilcoxon_holm": posthoc},
        "secondary_accuracy_tip100_vs_tip0": accuracy_test(wide, "tip_100", "tip_0"),
        "negative_control_length_aa": length_test(wide, "tip_0_repeat", "tip_0"),
        "negative_control_accuracy_aa": accuracy_test(wide, "tip_0_repeat", "tip_0"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    make_figures(long, wide, out_dir)

    print(f"[{stage}] n = {len(wide)} questions\n")
    print(descriptives(long).round(3).to_string(), "\n")
    for name, r in [("PRIMARY length tip_100 vs tip_0", primary), ("  without truncated", primary_no_trunc), ("A/A length tip_0_repeat vs tip_0", summary["negative_control_length_aa"])]:
        lo, hi = r["length_ratio_95ci"]
        print(f"{name:34s} ratio {r['length_ratio']:.3f} [{lo:.3f}, {hi:.3f}]  dz {r['cohens_dz']:+.3f}  t({r['df']}) = {r['t']:+.2f} p = {r['p_ttest']:.4g}  wilcoxon p = {r['p_wilcoxon']:.4g}  r_rb {r['rank_biserial_r']:+.3f}")
    print(f"\nFriedman none/tip_0/tip_100: chi2 = {fr.statistic:.2f}, p = {fr.pvalue:.4g}")
    for p in posthoc:
        print(f"  {p['comparison']:18s} ratio {p['length_ratio']:.3f}  wilcoxon p = {p['p_wilcoxon']:.4g}  Holm p = {p['p_wilcoxon_holm']:.4g}")
    for name, key in [("accuracy tip_100 vs tip_0", "secondary_accuracy_tip100_vs_tip0"), ("A/A accuracy", "negative_control_accuracy_aa")]:
        r = summary[key]
        lo, hi = r["difference_95ci"]
        print(f"\n{name}: diff {r['accuracy_difference']:+.3f} [{lo:+.3f}, {hi:+.3f}]  discordant {r['discordant_share']:.3f}  McNemar p = {r['p_mcnemar']:.4g}  TOST(+-{EQUIV_MARGIN}) p = {r['p_tost']:.4g}  equivalent = {r['equivalent_within_margin']}")
    print(f"\nWrote {out_dir / 'summary.json'} and figures")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=STAGES)
    analyze(parser.parse_args().stage)


if __name__ == "__main__":
    main()
