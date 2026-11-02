"""Extra figures for the report and presentation (main study).

Reads the graded main-study data and the pilot-based power settings, and writes five
figures to results/figures/:

  1. forest_plot.png        accuracy change with 95% CIs: primary, sensitivity, subgroups, A/A control
  2. power_curve.png        McNemar power at n = 1,219 vs. effect size, with the SESOI and observed effect
  3. null_distribution.png  what H0 predicts for the 132 discordant problems vs. the observed 99 : 33 split
  4. flip_breakdown.png     why the 99 right -> wrong problems failed (truncated / wrong answer)
  5. length_ratio_hist.png  per-problem change in answer length, distractor vs. original against A/A noise

All numbers come from the same functions as src/analyze.py and src/power_analysis.py,
so the figures always match results/main/summary.json.

Usage: python src/visualize.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from analyze import accuracy_test  # noqa: E402
from config import RESULTS, wide_path  # noqa: E402
from power_analysis import SESOI, mcnemar_power  # noqa: E402

OUT = RESULTS / "figures"

# Colors (validated categorical slots + neutral ink); identity is never carried by color alone.
BLUE = "#2a78d6"  # distractor vs original
ORANGE = "#eb6834"  # second series / highlights
AQUA = "#1baf7a"  # third series
GRAY = "#9a9994"  # A/A noise / reference
INK = "#0b0b0b"
INK_2 = "#52514e"

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.edgecolor": "#c9c8c3",
    "axes.labelcolor": INK_2,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#ecebe7",
    "grid.linewidth": 0.8,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "legend.frameon": False,
})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote results/figures/{name}")


# ---------------------------------------------------------------- 1. forest plot
def forest_plot(wide: pd.DataFrame) -> None:
    has_num = wide["has_distractor_number"].astype(bool)
    rows = [
        ("All problems (primary)", accuracy_test(wide, "distractor", "original"), BLUE),
        ("Without truncated answers", accuracy_test(wide, "distractor", "original", drop_truncated=True), BLUE),
        ("Distractor adds a new number", accuracy_test(wide[has_num], "distractor", "original"), BLUE),
        ("Distractor adds no new number", accuracy_test(wide[~has_num], "distractor", "original"), BLUE),
        ("A/A control (identical rerun)", accuracy_test(wide, "original_repeat", "original"), GRAY),
    ]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, (label, r, color) in enumerate(rows):
        y = len(rows) - 1 - i
        d, (lo, hi) = r["accuracy_difference"] * 100, [v * 100 for v in r["difference_95ci"]]
        ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round")
        ax.plot(d, y, "o", ms=8, color=color, mec="white", mew=1.5, zorder=3)
        p = r["p_mcnemar"]
        ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}" if p < 0.01 else f"p = {p:.2f}"
        ax.text(1.6, y, f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]   {ptxt}   n = {r['n']:,}",
                va="center", fontsize=9, color=INK_2)
    ax.axvline(0, color=INK_2, lw=1)
    ax.axvline(-SESOI * 100, color=ORANGE, lw=1.2, ls="--")
    ax.text(-SESOI * 100, len(rows) - 0.45, "smallest effect\nof interest (−2)", color=ORANGE,
            fontsize=8, ha="center", va="bottom", bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1])
    ax.set_xlim(-10, 1.5)
    ax.set_ylim(-0.6, len(rows) + 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Change in accuracy (percentage points, 95% CI)")
    ax.set_title("Accuracy drops in every subset; identical reruns do not move", loc="left")
    save(fig, "forest_plot.png")


# ---------------------------------------------------------------- 2. power curve
def power_curve(n: int, observed_delta: float, observed_disc: float) -> None:
    pa = json.loads((RESULTS / "power_analysis.json").read_text())
    p_pilot, p_upper = pa["pilot_discordant_share"], pa["pilot_discordant_share_upper80"]
    deltas = np.linspace(0.002, 0.07, 300)
    fig, ax = plt.subplots(figsize=(7, 4))
    for p_disc, color, ls, label in [(p_pilot, BLUE, "-", f"p_disc = {p_pilot:.2f} (pilot)"),
                                     (p_upper, ORANGE, "--", f"p_disc = {p_upper:.3f} (pilot 80% upper limit)")]:
        ax.plot(deltas * 100, [mcnemar_power(n, p_disc, d) for d in deltas], color=color, ls=ls, lw=2, label=label)
    ax.axhline(0.8, color=INK_2, lw=1, ls=":")
    ax.text(6.9, 0.81, "target power 0.80", ha="right", va="bottom", fontsize=8, color=INK_2)
    ax.axvline(SESOI * 100, color=GRAY, lw=1.2)
    ax.text(SESOI * 100 + 0.1, 0.05, "smallest effect\nof interest", fontsize=8, color=INK_2)
    ax.axvline(abs(observed_delta) * 100, color=INK, lw=1.2)
    ax.text(abs(observed_delta) * 100 - 0.1, 0.45, f"observed\n{abs(observed_delta) * 100:.1f} points", fontsize=8, color=INK, ha="right")
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("True accuracy difference δ (percentage points)")
    ax.set_ylabel("Power of McNemar's test")
    ax.set_title(f"Power at n = {n:,}: enough for ≥ 3-point effects", loc="left")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.62), fontsize=9)
    ax.text(0.01, -0.2, f"Observed discordant share in the main study: {observed_disc:.3f}. α = 0.05, two-sided; Connor (1987).",
            transform=ax.transAxes, fontsize=8, color=INK_2)
    save(fig, "power_curve.png")


# ---------------------------------------------------------------- 3. null distribution
def null_distribution(only_original: int, only_distractor: int, p_value: float) -> None:
    disc = only_original + only_distractor
    k = np.arange(disc + 1)
    pmf = stats.binom.pmf(k, disc, 0.5)
    extreme = (k >= only_original) | (k <= disc - only_original)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(k, pmf, width=0.9, color=GRAY, label="Possible splits if H₀ were true (about 66 : 66)")
    ax.axvspan(only_original - 0.5, disc, color=ORANGE, alpha=0.15, lw=0,
               label=f"As or more extreme than observed (both tails), total probability = p")
    ax.axvspan(0, disc - only_original + 0.5, color=ORANGE, alpha=0.15, lw=0)
    ax.axvline(only_original, color=INK, lw=1.5)
    ax.annotate(f"observed: {only_original} right → wrong\n({only_distractor} wrong → right)",
                xy=(only_original, 0.004), xytext=(only_original - 3, 0.045), ha="right", fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1))
    ax.text(0.99, 0.95, f"McNemar p = {p_value:.1e}", transform=ax.transAxes, ha="right", va="top", fontsize=10, color=INK)
    ax.set_xlim(20, disc - 10)
    ax.set_xlabel(f"Number of the {disc} discordant problems that went right → wrong")
    ax.set_ylabel("Probability under H₀")
    ax.set_title("If the sentence had no effect, a 99 : 33 split would almost never happen", loc="left")
    ax.set_ylim(0, pmf.max() * 1.35)
    ax.legend(loc="upper left", fontsize=8)
    save(fig, "null_distribution.png")


# ---------------------------------------------------------------- 4. flip breakdown
def flip_breakdown(wide: pd.DataFrame) -> None:
    c_o, c_d = wide["correct_original"].astype(bool), wide["correct_distractor"].astype(bool)
    t_d, t_o = wide["truncated_distractor"].astype(bool), wide["truncated_original"].astype(bool)
    used = wide["uses_number_distractor"].fillna(False).astype(bool)
    r2w = c_o & ~c_d
    w2r = ~c_o & c_d
    segs_r2w = [
        ("Truncated at 2,048 tokens", int((r2w & t_d).sum()), GRAY),
        ("Wrong answer, mentions the new number", int((r2w & ~t_d & used).sum()), ORANGE),
        ("Wrong answer, no new number mentioned", int((r2w & ~t_d & ~used).sum()), BLUE),
    ]
    w2r_trunc = int((w2r & t_o).sum())
    segs_w2r = [("Original was truncated", w2r_trunc, GRAY), ("Original had a wrong answer", int(w2r.sum()) - w2r_trunc, AQUA)]

    fig, ax = plt.subplots(figsize=(8, 2.8))
    for y, segs in [(1, segs_r2w), (0, segs_w2r)]:
        left = 0
        for label, v, color in segs:
            ax.barh(y, v, left=left, color=color, height=0.55, edgecolor="white", linewidth=2)
            if v >= 5:
                ax.text(left + v / 2, y, str(v), ha="center", va="center", color="white", fontsize=10, fontweight="bold")
            elif v:
                ax.text(left + v / 2, y + 0.4, str(v), ha="center", va="bottom", color=INK_2, fontsize=9)
            left += v
        ax.text(left + 1.5, y, f"{left} total", va="center", fontsize=9, color=INK_2)
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["Right → wrong\n(hurt by sentence)", "Wrong → right\n(helped by sentence)"])
    ax.set_xlim(0, 112)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Problems")
    legend = [("Truncated at 2,048 tokens", GRAY)] + [(lab, c) for lab, _, c in segs_r2w[1:]] + [("Original had a wrong answer", AQUA)]
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, c in legend]
    ax.legend(handles, [lab for lab, _ in legend], loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=8)
    ax.set_title("Most flips are reasoning errors, not running out of tokens", loc="left")
    save(fig, "flip_breakdown.png")


# ---------------------------------------------------------------- 5. length histogram
def length_hist(wide: pd.DataFrame) -> None:
    d_dis = np.log(wide["tokens_distractor"] / wide["tokens_original"])
    d_aa = np.log(wide["tokens_original_repeat"] / wide["tokens_original"])
    bins = np.linspace(-1, 1.5, 51)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.hist(d_aa.clip(-1, 1.5), bins=bins, color=GRAY, alpha=0.7, label="A/A noise: identical rerun vs. original")
    ax.hist(d_dis.clip(-1, 1.5), bins=bins, histtype="step", color=BLUE, lw=2, label="Distractor vs. original")
    ax.axvline(0, color=INK_2, lw=1)
    m = d_dis.mean()
    ax.axvline(m, color=BLUE, lw=1.2, ls="--")
    ax.text(m + 0.03, ax.get_ylim()[1] * 0.6, f"mean log ratio {m:+.3f}\n= {np.exp(m):.2f}× longer", color=BLUE, fontsize=9)
    ax.set_xlabel("log(answer tokens, condition / original)   ← shorter · longer →   (values beyond the range are shown at the edges)")
    ax.set_ylabel("Problems")
    ax.set_title("Answers get longer with the irrelevant sentence", loc="left")
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "length_ratio_hist.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wide = pd.read_csv(wide_path("main"))
    primary = accuracy_test(wide, "distractor", "original")
    forest_plot(wide)
    power_curve(primary["n"], primary["accuracy_difference"], primary["discordant_share"])
    null_distribution(primary["table"]["only_original"], primary["table"]["only_distractor"], primary["p_mcnemar"])
    flip_breakdown(wide)
    length_hist(wide)


if __name__ == "__main__":
    main()
