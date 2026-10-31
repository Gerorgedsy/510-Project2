# Results: Does Promising a Tip Make an LLM Write Longer Answers?

Main run on 2026-10-30. Numbers below come from `results/main/summary.json` (`python src/analyze.py main`).

**Bottom line:** a $100 tip made Qwen3.5-4B's answers slightly but reliably longer (+1.7%, p = 0.007). The effect is smaller than the 5% we had pre-specified as practically meaningful, and accuracy did not change: it was equivalent within ±2 points (TOST p = 0.001). The tip made the model say a little more, not answer better.

## Data collection

- 1,319 GSM8K test questions × 4 conditions = 5,276 calls; 0 failed requests.
- `Qwen/Qwen3.5-4B` on Tinker, greedy decoding, thinking off, max 2,048 output tokens; call order shuffled across conditions.
- 49 answers (0.9%) hit the token cap. They were spread evenly across conditions (12 / 13 / 12 / 12) and were the only parse failures; per the pre-registered rule they count as incorrect.

## Descriptive statistics

| Condition | Mean tokens | Median tokens | Geometric mean | Accuracy | Truncated |
|---|---|---|---|---|---|
| none | 402.9 | 352 | 364.1 | 0.930 | 12 |
| tip_0 | 400.0 | 351 | 362.1 | 0.926 | 13 |
| tip_100 | 405.0 | 357 | 368.2 | 0.930 | 12 |
| tip_0_repeat | 394.4 | 351 | 360.0 | 0.933 | 12 |

## Primary result: answer length, tip_100 vs tip_0

Per-question d = log(tokens_tip_100) − log(tokens_tip_0), n = 1,319.

| Measure | Value |
|---|---|
| Geometric-mean length ratio | **1.017**, 95% CI [1.004, 1.029] |
| Paired t-test | t(1318) = 2.69, **p = 0.0073** |
| Cohen's dz | 0.074 |
| Wilcoxon signed-rank | p = 2.6 × 10⁻¹², rank-biserial r = 0.25 |
| Direction | longer under the tip for 48.9% of questions, shorter for 28.1%, identical for 23.0% |

- **Decision:** reject H₀. With a tip, the model writes longer answers.
- **Size:** about +6 tokens on a ~365-token answer. The whole 95% CI lies below the pre-specified smallest effect of interest (+5%), so the effect is real but practically small. For comparison, the viral 2023 post (GPT-4, 5 runs per condition) reported +11%, and our pilot (n = 50) suggested +3.3%.
- **Assumptions:** d is strongly non-normal (Shapiro–Wilk W = 0.68, p < 0.001). It is peaked at 0, heavy-tailed, and skewed (−1.15) by a few truncated answers. With n = 1,319 the t-test relies on the central limit theorem, and the Wilcoxon test, which needs no normality, agrees. Its much smaller p-value reflects how consistent the direction is (49% longer vs. 28% shorter), even though the magnitudes are heavy-tailed.

### Sensitivity: without truncated answers

Dropping the 19 questions where either compared answer hit the cap (n = 1,300) gives a ratio of 1.019 [1.008, 1.030], t(1299) = 3.53, p = 0.0004, Wilcoxon p = 6.5 × 10⁻¹³. Skewness falls to −0.08, so the result is not driven by truncation.

## Secondary result: length across none / tip_0 / tip_100

Friedman test on log tokens: χ²(2) = 63.85, p = 1.4 × 10⁻¹⁴.

| Pairwise (Wilcoxon) | Length ratio | p | Holm-adjusted p |
|---|---|---|---|
| tip_100 vs tip_0 | 1.017 | 2.6 × 10⁻¹² | 7.7 × 10⁻¹² |
| tip_100 vs none | 1.011 | 2.5 × 10⁻⁴ | 5.0 × 10⁻⁴ |
| tip_0 vs none | 0.995 | 0.043 | 0.043 |

Answers order as tip_0 < none < tip_100: a $0 tip slightly shortens answers relative to no sentence, and a $100 tip lengthens them. The tip_0 vs none difference is only marginal (Holm p = 0.043), so we treat the "stingy tip" direction as suggestive.

## Secondary result: accuracy, tip_100 vs tip_0

| | Both correct | Only tip_100 correct | Only tip_0 correct | Both wrong |
|---|---|---|---|---|
| Questions | 1,200 | 27 | 22 | 70 |

| Measure | Value |
|---|---|
| Accuracy | 93.0% (tip_100) vs 92.6% (tip_0) |
| Difference | +0.4 points, 95% CI [−0.7, +1.4] |
| McNemar (χ², continuity-corrected) | p = 0.57 |
| Paired odds ratio | 1.23 |
| TOST, ±2-point margin | p = 0.001; 90% CI [−0.5, +1.3] points lies inside the margin, so **equivalent** |

- **Decision:** we cannot reject equal accuracy. More importantly, the TOST shows any accuracy effect is smaller than 2 points.
- **Per-question vs. aggregate:** the tip changed the outcome of 3.7% of questions (49 discordant pairs), about twice the A/A noise level (1.8%), but the changes went both ways and cancelled out. This mirrors Meincke et al. (2025): prompt variations shift individual questions without moving the aggregate score.

## Negative control (A/A): tip_0_repeat vs tip_0

The two prompts are identical, so any difference is server-side noise.

| Outcome | Result |
|---|---|
| Length | ratio 0.994 [0.986, 1.002], t(1318) = −1.45, p = 0.15; Wilcoxon p = 0.93; longer 17.7% / shorter 16.7% |
| Accuracy | +0.6 points [−0.1, +1.3], McNemar p = 0.15; 24 discordant pairs (1.8%) |

There is no systematic drift between identical runs: the shuffled call order kept server nondeterminism unbiased. (The pilot's A/A Wilcoxon p = 0.034 on 16 non-identical pairs was a chance result.) The tip effect on length is about twice the width of the A/A noise interval.

## Other observations

- The model essentially never acknowledges the tip. The `mentions_tip` rate (0.6–0.9%) is the same in the `none` condition, so those hits are GSM8K questions that are themselves about restaurant tips. This differs from DeepSeek in our exploratory runs, which often replied "No tip needed".

## Figures (`results/main/`)

- `length_ratio_hist.png`: distribution of per-question log length ratios, tip_100 vs tip_0 against the A/A noise distribution. Recommended for the presentation.
- `accuracy_by_condition.png`: accuracy per condition with Wilson 95% CIs.
- `tokens_by_condition.png`: geometric-mean tokens per condition. **Caution:** these are per-condition (unpaired) CIs. They overlap even though the paired test is significant, because the design is within-question; read the paired results above, not the overlap.

## Interpretation and limitations

- The folklore is directionally right about length but greatly overstates it: on a small modern model the tip effect is about +2%, not +11%. It brings no accuracy benefit.
- Statistical vs. practical significance: n = 1,319 detects a 1.7% length change with high confidence, but the pre-registered 5% threshold says this change is too small to matter in practice.
- These results cover one model (Qwen3.5-4B), one task (GSM8K with step-by-step reasoning), one tip wording at the end of the prompt, and two amounts. Accuracy is near ceiling (93%), and about 5% of GSM8K labels are known to be wrong, so small accuracy effects would be hard to see. The TOST still bounds them below 2 points.
