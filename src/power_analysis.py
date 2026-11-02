"""Power / sample-size analysis for McNemar's test.

Uses Connor's (1987) normal approximation for paired binary outcomes:

    n = [ z_{a/2} * sqrt(p_disc) + z_b * sqrt(p_disc - delta^2) ]^2 / delta^2

where
    p_disc = p10 + p01  proportion of discordant pairs
    delta  = p10 - p01  difference in accuracy between conditions

Assumptions (to be justified in the report):
  * two-sided alpha = 0.05, target power = 0.80
  * prior work suggests tipping effects are small -> delta of 2-5 points
  * proportion of discordant pairs 10-20% (to be checked with a pilot)
"""

from math import sqrt

from scipy.stats import norm

ALPHA = 0.05
POWER = 0.80
N_AVAILABLE = 1319  # size of GSM8K test set


def mcnemar_n(p_disc: float, delta: float, alpha: float = ALPHA, power: float = POWER) -> int:
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    n = (z_a * sqrt(p_disc) + z_b * sqrt(p_disc - delta**2)) ** 2 / delta**2
    return int(n) + 1


def mcnemar_power(n: int, p_disc: float, delta: float, alpha: float = ALPHA) -> float:
    z_a = norm.ppf(1 - alpha / 2)
    z_b = (sqrt(n) * abs(delta) - z_a * sqrt(p_disc)) / sqrt(p_disc - delta**2)
    return float(norm.cdf(z_b))


def main() -> None:
    print(f"alpha = {ALPHA}, target power = {POWER}, available n = {N_AVAILABLE}\n")
    print(f"{'p_disc':>7} {'delta':>6} {'n needed':>9} {'power @ n=1319':>15}")
    for p_disc in (0.10, 0.15, 0.20):
        for delta in (0.02, 0.03, 0.05):
            n = mcnemar_n(p_disc, delta)
            pw = mcnemar_power(N_AVAILABLE, p_disc, delta)
            print(f"{p_disc:>7.2f} {delta:>6.2f} {n:>9d} {pw:>15.2f}")


if __name__ == "__main__":
    main()
