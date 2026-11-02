# Results: Can One Irrelevant Sentence Make an LLM Fail a Math Problem?

Main run on 2026-10-30. Numbers below come from `results/main/summary.json` (`python src/analyze.py main`).

**Bottom line:** inserting one irrelevant sentence lowered Qwen3.5-4B's accuracy on GSM8K from 93.1% to 87.7%, −5.4 points (95% CI [−7.2, −3.6]), McNemar χ²(1) = 32.01, p = 1.5 × 10⁻⁸. Its error rate almost doubled, from 6.9% to 12.3%. The effect is far beyond the A/A noise of identical reruns (3 vs 3 flips), holds without truncated answers (−3.8 points, p = 3.0 × 10⁻⁵), and its entire CI lies beyond the pre-specified 2-point smallest effect of interest. It is both statistically and practically significant.

## Data collection

- 1,219 main-set problems (GSM8K test problems not used in the pilot) × 3 conditions = 3,657 calls; 0 failed requests.
- `Qwen/Qwen3.5-4B` on Tinker, greedy decoding, thinking off, max 2,048 output tokens; call order shuffled across conditions.
- 65 answers (1.8%) hit the token cap: 12 original, 39 distractor, 14 original_repeat. They were the only parse failures, and per the pre-registered rule they count as incorrect.

## Descriptive statistics

| Condition | Accuracy | Mean tokens | Median tokens | Geometric mean | Truncated |
|---|---|---|---|---|---|
| original | 0.931 | 405.6 | 353 | 366.9 | 12 |
| distractor | **0.877** | 504.6 | 408 | 440.0 | 39 |
| original_repeat | 0.931 | 410.1 | 352 | 368.6 | 14 |

## Primary result: accuracy, distractor vs original

| | Both correct | Only original correct (right → wrong) | Only distractor correct (wrong → right) | Both wrong |
|---|---|---|---|---|
| Problems | 1,036 | **99** | 33 | 51 |

| Measure | Value |
|---|---|
| Accuracy | 87.7% (distractor) vs 93.1% (original) |
| Paired difference | **−5.4 points**, 95% CI [−7.2, −3.6] |
| McNemar (χ², continuity-corrected) | χ²(1) = 32.01, **p = 1.5 × 10⁻⁸** |
| Exact McNemar (binomial, robustness) | p = 7.5 × 10⁻⁹ |
| Conditional odds ratio (wrong → right / right → wrong) | **0.33**, exact 95% CI [0.22, 0.50] |

- **Decision:** reject H₀. With the irrelevant sentence, the model answers correctly less often.
- **Size:** of the 132 problems whose correctness changed, 99 (75%) went right → wrong and only 33 went wrong → right, a 3 : 1 ratio. Under H₀ the split would be 50 : 50. In absolute terms, 66 more problems were answered wrongly out of 1,219.
- **Practical significance:** the whole 95% CI lies beyond the pre-specified smallest effect of interest (2 points).
- **Assumptions:** McNemar's test needs paired binary outcomes and independent pairs. Each problem gives one pair, and problems are distinct GSM8K items answered in separate calls. The χ² approximation needs enough discordant pairs: there are 132, well above the pre-registered cut-off of 25, and the exact binomial version gives the same conclusion.
- **Power check:** the observed discordant share was 10.8%, close to the pilot's 10% and below the conservative 13.4% used for planning. The observed effect (5.4 points) is larger than the pilot estimate (4 points), at which the planned power was 0.99.

### Sensitivity: without truncated answers

The distractor made answers run out of tokens more often. Of the 47 problems where either answer hit the cap, 35 were truncated only under the distractor, 8 only under the original, and 4 under both. 24 of the 99 right → wrong flips are truncations. Dropping those 47 problems (n = 1,172):

| Measure | Value |
|---|---|
| Accuracy | 91.0% (distractor) vs 94.8% (original) |
| Table | 1,036 both correct / 75 right → wrong / 31 wrong → right / 30 both wrong |
| Paired difference | −3.8 points, 95% CI [−5.5, −2.0] |
| McNemar | χ²(1) = 17.44, p = 3.0 × 10⁻⁵; OR 0.41 [0.26, 0.64] |

The effect shrinks but stays significant: roughly 70% of it remains once truncation is removed, and the rest comes from the distractor pushing answers past the token limit.

## Negative control (A/A): original_repeat vs original

The two prompts are identical, so any difference is server-side noise.

| Outcome | Result |
|---|---|
| Accuracy | 93.1% vs 93.1%; 3 vs 3 discordant pairs (0.5%); difference 0.0 [−0.4, +0.4] points; exact McNemar p = 1.0 |
| Length | ratio 1.005 [0.998, 1.012], t(1218) = 1.32, p = 0.19; Wilcoxon p = 0.34 |

Server nondeterminism changed the correctness of only 6 problems, with no direction. The distractor changed 132, mostly in one direction. The accuracy drop is not noise.

## Secondary result: answer length, distractor vs original

Per-problem d = log(tokens_distractor) − log(tokens_original), n = 1,219.

| Measure | Value |
|---|---|
| Geometric-mean length ratio | 1.199, 95% CI [1.172, 1.226] |
| Paired t-test | t(1218) = 15.84, p = 1.6 × 10⁻⁵¹; Cohen's dz = 0.45 |
| Wilcoxon signed-rank | p = 9.3 × 10⁻⁸⁰, rank-biserial r = 0.63 |
| Direction | longer for 75.2% of problems, shorter for 24.1% |

- Answers were about 20% longer with the distractor (median +45 tokens), partly because the problem text is longer and partly because the model often discusses the extra sentence.
- **Assumptions:** d is non-normal (Shapiro–Wilk W = 0.84, p < 0.001; skew 1.01). With n = 1,219 the t-test relies on the central limit theorem, and the Wilcoxon test agrees. Without truncated answers: ratio 1.167 [1.146, 1.189], p = 8.1 × 10⁻⁵⁵.

## Exploratory: does the model use the distractor's number?

985 of the 1,219 problems gain a new digit-number from the inserted sentence.

- The distractor answer mentions that number in **83%** of these problems. The original answer, which never saw it, contains the same number by coincidence in 38%.
- 77 of the 84 right → wrong flips in this subset mention the new number.
- But mentioning is not the same as misusing. Among these 985 distractor answers, mentioning the number was not significantly linked to a wrong answer (Fisher's exact OR = 1.48, p = 0.20), because most correct answers mention it too, often to dismiss it.
- The 234 problems whose distractor adds no new number also lost accuracy: 95.3% → 89.7%, with 15 right → wrong vs 2 wrong → right. So the effect is not only about stray numbers.

Examples from the main run (right with the original, wrong with the distractor):

- **#261**, added "and 500 nickels". The model added the nickels' $25 to the pennies' $62 and answered 87 (reference 62).
- **#733**, added "and a $5 bag of lollipops". The model split $25 instead of $20 and answered 6.25 (reference 5).
- **#652**, added "He also has a $10 rental voucher but finds it expired". The model correctly ignored the voucher, yet now misread "each hour after that is twice the cost" and answered 105 (reference 75). It had solved the original version correctly.

## Correction to the grading code

The pre-registered rule (README, `src/grade.py` docstring) counts truncated answers as incorrect. The first version of `grade.py` applied this only when no number could be parsed, so a truncated answer could still count as correct if it happened to contain the right `Answer:` number. We found this after the first main analysis, fixed the code to match the rule, and re-graded. The fix affected 2 truncated distractor answers. Before the fix: right → wrong 97, difference −5.3 points, p = 3.3 × 10⁻⁸. After the fix: right → wrong 99, difference −5.4 points, p = 1.5 × 10⁻⁸. The conclusion is unchanged, and the sensitivity analysis is unaffected because it drops truncated answers anyway. The pilot was also unaffected: its truncated answers had no parsable number.

## Figures (`results/main/`)

- `discordant_pairs.png`: right → wrong vs wrong → right counts, distractor vs original against the A/A control. Recommended for the presentation: 99 vs 33 next to 3 vs 3 shows the effect and the noise floor in one picture.
- `accuracy_by_condition.png`: accuracy per condition with Wilson 95% CIs. **Caution:** these are per-condition (unpaired) CIs. Read the paired McNemar test above for the comparison, not the overlap of the bars.

## Interpretation and limitations

- One irrelevant but topic-related sentence nearly doubles a small modern model's error rate on grade-school math. This replicates, with a paired and pre-registered design, the distractibility reported by Shi et al. (2023), Mirzadeh et al. (2024), and Li et al. (2024).
- The drop has two routes: wrong reasoning (the model folds the extra information into the calculation, or reasons differently even when it dismisses it) and running out of tokens. The effect remains significant when the second route is removed.
- Scope: one model (Qwen3.5-4B, thinking off), one task (GSM8K), one perturbation type (GSM-Plus "distraction insertion", written by the GSM-Plus authors). Larger models or thinking mode may be more robust.
- Some distractors may be arguably relevant or ambiguous (e.g. #733: is a "$5 bag of lollipops" part of "the money"?). We did not audit all 1,219 by hand, so part of the effect may reflect ambiguous wording rather than pure distraction.
- Baseline accuracy is high (93%), and about 5% of GSM8K labels are known to be wrong (Vendrow et al., 2025), which limits how many problems can flip. The distractor-number measure detects only numbers written as digits and counts mentions, not use.

## Suggested reporting line

> Inserting one irrelevant sentence reduced accuracy from 93.1% to 87.7% (paired difference −5.4 percentage points, 95% CI [−7.2, −3.6]; McNemar's χ²(1) = 32.01, p < .001; conditional OR = 0.33, 95% CI [0.22, 0.50]; n = 1,219 problems).
