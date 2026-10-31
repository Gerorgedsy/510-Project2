"""Sample-size justification, computed from the pilot.

Primary outcome: per-question d = log(tokens_tip_100) - log(tokens_tip_0), two-sided paired
t-test at alpha = 0.05, target power 0.80.
  * Smallest effect size of interest (SESOI): a 5% change in answer length, |d| = log(1.05).
  * SD of d from the pilot, made conservative by using its one-sided 80% upper confidence
    limit (Browne 1995); dz = SESOI / SD.
  * n is divided by 0.864 (Wilcoxon's asymptotic relative efficiency) so the Wilcoxon
    robustness test is powered too.
Secondary outcome: accuracy, McNemar's test (Connor 1987 normal approximation) using the
pilot's discordant-pair rate.

The main stage uses all 1,319 GSM8K test questions; this script checks that n is enough.

Usage: python src/power_analysis.py      (needs data/processed/pilot_wide.csv)
Writes results/power_analysis.json
"""

import json
import math

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestPower

from config import RESULTS, wide_path

ALPHA = 0.05
POWER = 0.80
SESOI = math.log(1.05)
WILCOXON_ARE = 0.864
N_MAIN = 1319


def mcnemar_power(n: int, p_disc: float, delta: float, alpha: float = ALPHA) -> float:
    """Connor (1987) approximate power of McNemar's test."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = (math.sqrt(n) * abs(delta) - z_a * math.sqrt(p_disc)) / math.sqrt(p_disc - delta**2)
    return float(stats.norm.cdf(z_b))


def main() -> None:
    wide = pd.read_csv(wide_path("pilot"))
    n_pilot = len(wide)
    d = np.log(wide["tokens_tip_100"]) - np.log(wide["tokens_tip_0"])
    d_aa = np.log(wide["tokens_tip_0_repeat"]) - np.log(wide["tokens_tip_0"])

    sd = d.std(ddof=1)
    sd_ucl80 = sd * math.sqrt((n_pilot - 1) / stats.chi2.ppf(0.20, n_pilot - 1))
    dz_plan = SESOI / sd_ucl80
    tp = TTestPower()
    n_t = math.ceil(tp.solve_power(effect_size=dz_plan, alpha=ALPHA, power=POWER))
    n_w = math.ceil(n_t / WILCOXON_ARE)
    power_main = tp.power(effect_size=dz_plan, nobs=N_MAIN, alpha=ALPHA)
    dz_obs = d.mean() / sd
    power_main_obs = tp.power(effect_size=abs(dz_obs), nobs=N_MAIN, alpha=ALPHA)

    tip, no_tip = wide["correct_tip_100"], wide["correct_tip_0"]
    p_disc_pilot = float(((tip & ~no_tip) | (~tip & no_tip)).mean())
    acc_rows = []
    for p_disc in sorted({round(p_disc_pilot, 3), 0.05, 0.10}):
        for delta in (0.02, 0.03, 0.05):
            if p_disc > delta**2:
                acc_rows.append({"p_disc": p_disc, "delta": delta, "power_at_n_main": round(mcnemar_power(N_MAIN, p_disc, delta), 3)})

    summary = {
        "primary_length": {
            "n_pilot": n_pilot,
            "alpha": ALPHA,
            "target_power": POWER,
            "sesoi_ratio": 1.05,
            "sesoi_log": SESOI,
            "pilot_mean_log_ratio": float(d.mean()),
            "pilot_sd_log_ratio": float(sd),
            "sd_upper80_used_for_planning": float(sd_ucl80),
            "aa_noise_sd_log_ratio": float(d_aa.std(ddof=1)),
            "dz_planning": float(dz_plan),
            "n_required_paired_t": n_t,
            "n_required_with_wilcoxon_margin": n_w,
            "n_main": N_MAIN,
            "power_at_n_main_for_sesoi": float(power_main),
            "pilot_observed_dz": float(dz_obs),
            "power_at_n_main_for_pilot_effect": float(power_main_obs),
        },
        "secondary_accuracy_mcnemar": {"pilot_discordant_share": p_disc_pilot, "scenarios": acc_rows},
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "power_analysis.json").write_text(json.dumps(summary, indent=2))

    p = summary["primary_length"]
    print(f"Primary (length): pilot n = {n_pilot}, mean log ratio = {p['pilot_mean_log_ratio']:+.3f}, SD = {sd:.3f} (80% UCL {sd_ucl80:.3f}); A/A noise SD = {p['aa_noise_sd_log_ratio']:.3f}")
    print(f"  SESOI 5% -> dz = {dz_plan:.3f}; n needed = {n_t} (paired t), {n_w} (with Wilcoxon margin)")
    print(f"  power at n = {N_MAIN}: {power_main:.3f} for the SESOI, {power_main_obs:.3f} for the pilot's observed dz = {dz_obs:+.3f}")
    print(f"Secondary (accuracy): pilot discordant share = {p_disc_pilot:.3f}")
    print(pd.DataFrame(acc_rows).to_string(index=False))


if __name__ == "__main__":
    main()
