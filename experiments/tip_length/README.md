# Does Promising a Tip Make an LLM Write Longer Answers?

A self-contained experiment: everything it needs lives in this folder. The only file it reads from outside is the repo-root `.env` (`TINKER_API_KEY`).

## Motivation

A viral 2023 post claimed ChatGPT writes longer answers when promised a tip (five runs per condition, no statistical test). Later studies looked at **accuracy** and found little or no effect on large models (Meincke et al., 2025), while a tip helped a small 7B model on classification tasks (Salinas & Morstatter, 2024, also without a significance test). Nobody has checked the original **length** claim with a powered, pre-registered design. We test it on a small modern model, with accuracy as a secondary outcome: does a tip make the model *say more*, and does it make it *answer better*?

## Hypotheses

**Primary (answer length), tip_100 vs tip_0**
- **H₀:** the mean per-question log ratio of answer lengths, log(tokens with $100 tip / tokens with $0 tip), is 0.
- **H₁:** it is not 0 (two-sided, α = 0.05).

**Secondary**
- Length differs across `none` / `tip_0` / `tip_100` (Friedman test, Holm-corrected pairwise Wilcoxon post hoc).
- Accuracy: H₀ the probability of a correct answer is the same under tip_100 and tip_0 (McNemar). We also test whether the two are *equivalent* within ±2 accuracy points (TOST), so a null result can be read as "no meaningful effect" rather than "no evidence".

**Negative control (A/A):** `tip_0_repeat` is a word-for-word copy of `tip_0`. Any difference between the two is server-side noise, which gives a baseline for judging the tip effect.

## Design

| Element | Choice |
|---|---|
| Design | Paired (within-question): every question is answered under all four conditions |
| Questions | Main: all 1,319 GSM8K test questions. Pilot: 50 random GSM8K **train** questions (no overlap) |
| Independent variable | Sentence appended to the prompt (see conditions) |
| Primary outcome | Answer length in generated tokens (analysed on the log scale) |
| Secondary outcomes | Correct / incorrect (final `Answer: <number>` vs. GSM8K reference); whether the answer mentions the tip |
| Model | `Qwen/Qwen3.5-4B` via [Tinker](https://thinkingmachines.ai/tinker/), chat template applied, thinking mode off |
| Decoding | Greedy (temperature 0), max 2,048 output tokens, same for every condition |
| Randomization | All (question, condition) calls run in one shuffled order (seed 510), so time-of-day or server-load drift cannot line up with a condition |
| Grading | Rule fixed on pilot data before the main run (see `src/grade.py`); unparseable or truncated answers count as incorrect |

### Conditions

Prompt: *"Solve the following math word problem. Show your reasoning step by step, then write the final numeric answer on the last line in the form 'Answer: <number>'.* `Question: …`" followed by:

| Condition | Appended sentence | Role |
|---|---|---|
| `none` | *(nothing)* | plain baseline |
| `tip_0` | "I'm going to tip $0 for a perfect answer!" | control, same wording as the tip |
| `tip_100` | "I'm going to tip $100 for a perfect answer!" | treatment |
| `tip_0_repeat` | same as `tip_0` | A/A negative control |

`tip_0` and `tip_100` differ only in the dollar amount, so the primary comparison isolates the money from "an extra sentence was added".

## Statistical analysis plan (fixed before the main run)

- **Primary:** per question d = log(tokens_tip_100) − log(tokens_tip_0). Two-sided paired t-test (α = 0.05); report the geometric-mean length ratio exp(mean d) with a 95% CI and Cohen's dz.
- **Assumption checks:** Shapiro–Wilk and skewness of d; with n = 1,319 the t-test is robust to non-normality, and the **Wilcoxon signed-rank test** (with rank-biserial r) is reported as a robustness check.
- **Sensitivity:** repeat the primary test without questions where either answer hit the token cap.
- **Secondary length:** Friedman test on log tokens across none / tip_0 / tip_100, then pairwise Wilcoxon with Holm correction.
- **Secondary accuracy:** McNemar's test (exact if fewer than 25 discordant pairs), paired accuracy difference with 95% CI, paired odds ratio, and TOST equivalence with a ±2-point margin (equivalent if the 90% CI lies inside the margin).
- **Negative control:** the same length and accuracy tests on tip_0_repeat vs tip_0.

## Power analysis (`src/power_analysis.py`, from the pilot)

- **Smallest effect size of interest:** a 5% change in answer length (|d| = log 1.05 = 0.049).
- **Pilot SD of d:** 0.145. For planning we use its one-sided 80% upper confidence limit, 0.160 (Browne, 1995), so dz = 0.305.
- **Required n:** 87 questions for the paired t-test at α = 0.05 and power 0.80, or 101 after dividing by Wilcoxon's relative efficiency (0.864).
- **Main study n = 1,319**, so power is above 0.99 for the 5% effect.
- **Accuracy:** the pilot had 0 of 50 discordant pairs. If 5% of pairs are discordant, n = 1,319 gives 0.90 power to detect a 2-point accuracy difference.

## Pilot results (planning only; not part of the confirmatory analysis)

50 train questions × 4 conditions; no failed calls, no parse failures, no truncation.

| Condition | Mean tokens | Accuracy |
|---|---|---|
| none | 387.3 | 0.92 |
| tip_0 | 384.9 | 0.96 |
| tip_100 | 400.8 | 0.96 |
| tip_0_repeat | 385.3 | 0.96 |

- **tip_100 vs tip_0:** length ratio 1.033, 95% CI [0.991, 1.076], p = 0.12. That is about 3% longer, not significant at n = 50, as expected.
- **A/A noise:** 34 of 50 tip_0 / tip_0_repeat pairs had identical length. Server nondeterminism is real but small: the SD of d is 0.046, against 0.145 for the tip comparison.
- **Tip mentions:** the model never mentioned the tip.

## How to reproduce

```bash
cd experiments/tip_length
uv venv .venv --python 3.13 && uv pip install --python .venv/bin/python -r requirements.txt
# put TINKER_API_KEY=<your key> in the repo-root .env

.venv/bin/python src/load_data.py          # GSM8K -> data/raw/{pilot,main}_questions.csv
.venv/bin/python src/collect.py pilot      # 200 calls, ~30 s
.venv/bin/python src/grade.py pilot
.venv/bin/python src/power_analysis.py     # -> results/power_analysis.json
.venv/bin/python src/analyze.py pilot      # -> results/pilot/

.venv/bin/python src/collect.py main       # 5,276 calls, ~15 min, ~$2.3 on Tinker; safe to rerun if interrupted
.venv/bin/python src/grade.py main
.venv/bin/python src/analyze.py main       # -> results/main/
```

Tinker is not bit-for-bit deterministic even at temperature 0, so a rerun will reproduce the statistics but not every token. All raw outputs are saved in `data/raw/*_outputs.jsonl`.

## Folder structure

```
tip_length/
├── README.md
├── requirements.txt          # pinned versions
├── src/
│   ├── config.py             # model, decoding, seeds, file paths
│   ├── prompts.py            # template + the four conditions
│   ├── load_data.py          # GSM8K -> question sets
│   ├── sampler.py            # Tinker client (chat template, thinking off)
│   ├── collect.py            # async, resumable data collection
│   ├── grade.py              # answer extraction, correctness, length measures
│   ├── power_analysis.py     # sample-size justification from the pilot
│   └── analyze.py            # all tests + figures
├── data/
│   ├── raw/                  # questions + raw model outputs (jsonl)
│   └── processed/            # graded long and wide tables
└── results/
    ├── power_analysis.json
    ├── pilot/                # summary.json + figures
    └── main/
```

## Planned limitations

- One small model and one task type (math word problems with step-by-step reasoning).
- Token count measures verbosity, not quality; accuracy is the only quality check.
- GSM8K is popular and the model has likely seen it in training; about 5% of GSM8K labels are wrong (Vendrow et al., 2025), which caps measurable accuracy.
- The tip wording, its position (end of prompt) and the amounts ($0 / $100) are single choices.

## References

- Browne, R. H. (1995). On the use of a pilot sample for sample size determination. *Statistics in Medicine*, 14(17), 1933–1940.
- Cobbe, K., et al. (2021). Training verifiers to solve math word problems. arXiv:2110.14168.
- Lakens, D. (2017). Equivalence tests: A practical primer for t tests, correlations, and meta-analyses. *Social Psychological and Personality Science*, 8(4), 355–362.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153–157.
- Meincke, L., Mollick, E., Mollick, L., & Shapiro, D. (2025). Prompting Science Report 3: I'll pay you or I'll kill you — but will you care? arXiv:2508.00614.
- Salinas, A., & Morstatter, F. (2024). The butterfly effect of altering prompts: How small changes and jailbreaks affect large language model performance. arXiv:2401.03729.
- Vendrow, J., et al. (2025). Do large language model benchmarks test reliability? arXiv:2502.03461.
