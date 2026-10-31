# Can One Irrelevant Sentence Make an LLM Fail a Math Problem?

A self-contained experiment: everything it needs lives in this folder. The only file it reads from outside is the repo-root `.env` (`TINKER_API_KEY`).

## Motivation

Math word problems in the real world come with extra details that do not matter. A model that reasons, rather than pattern-matching on every number it sees, should ignore them. Earlier work found that large language models are easily distracted by irrelevant context (Shi et al., 2023; Mirzadeh et al., 2024), and the GSM-Plus benchmark (Li et al., 2024) includes a "distraction insertion" version of every GSM8K test problem. We test, with a paired and pre-registered design, whether one inserted, topic-related but irrelevant sentence lowers a small modern model's accuracy.

## Hypotheses

**Primary (accuracy), distractor vs original**
- **H₀:** the probability of a correct answer is the same with and without the irrelevant sentence. Equivalently, among problems where the two versions disagree, "right only without the sentence" and "right only with it" are equally likely.
- **H₁:** the probabilities differ (two-sided, α = 0.05).

**Secondary:** answer length differs between the two versions (paired t-test on log tokens, Wilcoxon as robustness).

**Exploratory:** when the inserted sentence adds a new number, how often the answer uses it (against a coincidence baseline), and whether using it goes with wrong answers.

**Negative control (A/A):** `original_repeat` is a word-for-word copy of `original`; any difference between them is server-side noise.

## Design

| Element | Choice |
|---|---|
| Design | Paired (within-problem): each problem is answered as original, with distractor, and as an A/A repeat |
| Problems | 1,319 GSM8K test problems matched to their GSM-Plus "distraction insertion" versions (exact seed-text match, 1,319/1,319). Pilot: 100 random problems. Main: the other 1,219 (no overlap) |
| Ground truth | GSM8K reference answer for both versions (GSM-Plus leaves the answer unchanged for this perturbation: 1,319/1,319 identical) |
| Independent variable | Problem text: original vs. with the inserted irrelevant sentence |
| Primary outcome | Correct / incorrect (`Answer: <number>` vs. reference) |
| Secondary outcomes | Answer length in tokens; whether the answer mentions the distractor's new number |
| Model | `Qwen/Qwen3.5-4B` via [Tinker](https://thinkingmachines.ai/tinker/), chat template applied, thinking mode off |
| Decoding | Greedy (temperature 0), max 2,048 output tokens, same for every condition |
| Randomization | All (problem, condition) calls run in one shuffled order (seed 510) |
| Grading | Rule fixed on pilot data before the main run (see `src/grade.py`); unparseable or truncated answers count as incorrect |

Prompt for every condition: *"Solve the following math word problem. Show your reasoning step by step, then write the final numeric answer on the last line in the form 'Answer: <number>'.* `Question: <problem text>`"

Example (GSM8K test #1, from GSM-Plus):
> **Original:** Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
>
> **Distractor:** … bakes muffins for her friends every day with four. ***She uses the seven eggshells for painting.*** She sells the remainder …

## Statistical analysis plan (fixed before the main run)

- **Primary:** McNemar's test on the paired 2×2 table (χ² with continuity correction; exact binomial if fewer than 25 discordant pairs). Report the paired accuracy difference with a 95% CI and the conditional odds ratio (only-original-right / only-distractor-right) with an exact 95% CI.
- **Sensitivity:** the same test without problems where either answer hit the token cap.
- **Secondary:** d = log(tokens_distractor) − log(tokens_original); paired t-test with geometric-mean length ratio and 95% CI; Wilcoxon signed-rank as robustness.
- **Exploratory:** share of distractor answers containing the new number, compared with the share of original answers that contain it by coincidence; Fisher's exact test of number use × wrong answer.
- **Negative control:** the same accuracy and length tests on original_repeat vs original.

## Power analysis (`src/power_analysis.py`, from the pilot)

McNemar power (Connor, 1987) at α = 0.05, smallest effect of interest a 2-point accuracy change.

- The pilot's discordant-pair rate was 10% (80% upper limit 13.4%).
- With main n = 1,219, power is **0.99 to detect a 4-point drop** (the pilot's estimate), 0.91 for 3 points (0.82 at the conservative discordance), and 0.60 for 2 points.
- The study is therefore well powered for effects of 3+ points, and underpowered only near the 2-point minimum.

## Pilot results (planning only; not part of the confirmatory analysis)

100 problems × 3 conditions = 300 calls, 0 failed; 4 truncated answers (2 distractor, 2 original_repeat), which were the only parse failures.

| Condition | Accuracy | Mean tokens |
|---|---|---|
| original | 0.94 | 366 |
| distractor | 0.90 | 480 |
| original_repeat | 0.94 | 389 |

- **Accuracy:** 7 problems went right → wrong with the distractor and 3 went wrong → right. That is −4.0 points, 95% CI [−10.1, +2.1], exact McNemar p = 0.34, as expected for n = 100. Without truncated answers: −2.0 points, p = 0.73.
- **A/A:** 0 discordant pairs.
- **Length:** distractor answers were **24% longer**, ratio 1.241 [1.141, 1.349], t-test p = 1.4 × 10⁻⁶, Wilcoxon p = 3.7 × 10⁻⁹. The A/A ratio was 1.017, p = 0.14.
- **Distractor numbers:** 90 of 100 pilot problems (1,075 of 1,319 overall) gain a new digit-number. It appears in 71% of distractor answers against 38% of original answers by coincidence. 6 of the 7 right → wrong flips mention it.
- **"Mentions" is not "uses":** in one flip the model explicitly dismissed the distractor and still made an unrelated counting slip.

## How to reproduce

```bash
cd experiments/irrelevant_sentence
uv venv .venv --python 3.13 && uv pip install --python .venv/bin/python -r requirements.txt
# put TINKER_API_KEY=<your key> in the repo-root .env

.venv/bin/python src/load_data.py          # GSM8K + GSM-Plus -> data/raw/{pilot,main}_questions.csv
.venv/bin/python src/collect.py pilot      # 300 calls, ~40 s
.venv/bin/python src/grade.py pilot
.venv/bin/python src/power_analysis.py     # -> results/power_analysis.json
.venv/bin/python src/analyze.py pilot      # -> results/pilot/

.venv/bin/python src/collect.py main       # 3,657 calls, ~10 min, ~$1.7 on Tinker; safe to rerun if interrupted
.venv/bin/python src/grade.py main
.venv/bin/python src/analyze.py main       # -> results/main/
```

Tinker is not bit-for-bit deterministic even at temperature 0, so a rerun will reproduce the statistics but not every token. All raw outputs are saved in `data/raw/*_outputs.jsonl`.

## Folder structure

```
irrelevant_sentence/
├── README.md
├── requirements.txt          # pinned versions
├── src/
│   ├── config.py             # model, decoding, seeds, file paths
│   ├── prompts.py            # template + the three conditions
│   ├── load_data.py          # GSM8K x GSM-Plus pairing -> question sets
│   ├── sampler.py            # Tinker client (chat template, thinking off)
│   ├── collect.py            # async, resumable data collection
│   ├── grade.py              # answer extraction, correctness, length, distractor-number use
│   ├── power_analysis.py     # McNemar sample-size justification from the pilot
│   └── analyze.py            # all tests + figures
├── data/
│   ├── raw/                  # question pairs + raw model outputs (jsonl)
│   └── processed/            # graded long and wide tables
└── results/
    ├── power_analysis.json
    ├── pilot/                # summary.json + figures
    └── main/
```

## Planned limitations

- One small model and one perturbation type. GSM-Plus distractors were written by its authors, and a few may be arguably relevant or ambiguous: in pilot problem #329, "5 white rabbit lollies" may or may not count as "lollipops".
- Baseline accuracy is high (~94%), and about 5% of GSM8K labels are wrong (Vendrow et al., 2025), which limits how many problems can flip.
- The distractor-number measure only detects numbers written as digits, and it counts mentions, including ones the model explicitly dismisses.

## References

- Cobbe, K., et al. (2021). Training verifiers to solve math word problems. arXiv:2110.14168.
- Connor, R. J. (1987). Sample size for testing differences in proportions for the paired-sample design. *Biometrics*, 43(1), 207–211.
- Li, Q., et al. (2024). GSM-Plus: A comprehensive benchmark for evaluating the robustness of LLMs as mathematical problem solvers. arXiv:2402.19255.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153–157.
- Mirzadeh, I., et al. (2024). GSM-Symbolic: Understanding the limitations of mathematical reasoning in large language models. arXiv:2410.05229.
- Shi, F., et al. (2023). Large language models can be easily distracted by irrelevant context. *ICML 2023*. arXiv:2302.00093.
- Vendrow, J., et al. (2025). Do large language model benchmarks test reliability? arXiv:2502.03461.
