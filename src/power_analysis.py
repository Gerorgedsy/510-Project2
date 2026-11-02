"""Sample-size justification for the primary accuracy test, computed from the pilot.

Primary outcome: correct / incorrect on the same problem with and without the irrelevant
sentence, tested with McNemar's test (two-sided, alpha = 0.05, target power 0.80), using
Connor's (1987) normal approximation:

    n = [ z_{a/2} * sqrt(p_disc) + z_b * sqrt(p_disc - delta^2) ]^2 / delta^2

  p_disc = share of discordant pairs (one version right, the other wrong)
  delta  = accuracy(distractor) - accuracy(original)

  * Smallest effect size of interest: a 2-point accuracy change (|delta| = 0.02).
  * p_disc from the pilot, made conservative with its one-sided 80% upper Clopper-Pearson limit.

Usage: python src/power_analysis.py      (needs data/processed/pilot_wide.csv)
Writes results/power_analysis.json
"""

import json
import math

import pandas as pd
from scipy import stats

from config import RESULTS, questions_path, wide_path

ALPHA = 0.05
POWER = 0.80
SESOI = 0.02


def mcnemar_n(p_disc: float, delta: float) -> int:
    z_a, z_b = stats.norm.ppf(1 - ALPHA / 2), stats.norm.ppf(POWER)
    return math.ceil((z_a * math.sqrt(p_disc) + z_b * math.sqrt(p_disc - delta**2)) ** 2 / delta**2)


def mcnemar_power(n: int, p_disc: float, delta: float) -> float:
    z_a = stats.norm.ppf(1 - ALPHA / 2)
    z_b = (math.sqrt(n) * abs(delta) - z_a * math.sqrt(p_disc)) / math.sqrt(p_disc - delta**2)
    return float(stats.norm.cdf(z_b))


def main() -> None:
    wide = pd.read_csv(wide_path("pilot"))
    n_main = len(pd.read_csv(questions_path("main")))
    x, y = wide["correct_original"], wide["correct_distractor"]
    n = len(wide)
    only_original, only_distractor = int((x & ~y).sum()), int((~x & y).sum())
    disc = only_original + only_distractor
    p_disc = disc / n
    p_disc_ucl80 = float(stats.beta.ppf(0.80, disc + 1, n - disc)) if disc < n else 1.0
    delta_obs = (only_distractor - only_original) / n

    scenarios = []
    for label, pd_ in [("pilot", p_disc), ("pilot 80% upper limit", p_disc_ucl80)]:
        for delta in sorted({SESOI, 0.03, 0.05, round(abs(delta_obs), 3)}):
            if delta > 0 and pd_ > delta**2:
                scenarios.append(
                    {"p_disc_source": label, "p_disc": round(pd_, 4), "delta": delta, "n_required": mcnemar_n(pd_, delta), "power_at_n_main": round(mcnemar_power(n_main, pd_, delta), 3)}
                )

    summary = {
        "n_pilot": n,
        "n_main": n_main,
        "alpha": ALPHA,
        "target_power": POWER,
        "sesoi_accuracy_points": SESOI,
        "pilot_accuracy_original": float(x.mean()),
        "pilot_accuracy_distractor": float(y.mean()),
        "pilot_only_original_correct": only_original,
        "pilot_only_distractor_correct": only_distractor,
        "pilot_discordant_share": p_disc,
        "pilot_discordant_share_upper80": p_disc_ucl80,
        "pilot_delta": delta_obs,
        "scenarios": scenarios,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "power_analysis.json").write_text(json.dumps(summary, indent=2))

    print(f"Pilot n = {n}: accuracy original {x.mean():.3f}, distractor {y.mean():.3f} (delta {delta_obs:+.3f})")
    print(f"discordant pairs: only original correct {only_original}, only distractor correct {only_distractor}; p_disc = {p_disc:.3f} (80% UCL {p_disc_ucl80:.3f})")
    print(f"main n = {n_main}")
    print(pd.DataFrame(scenarios).to_string(index=False))


if __name__ == "__main__":
    main()
