# Does Promising a Tip Make an LLM Better at Math?

**AIPI 510 – Sourcing Data for Analytics · Module 2 Project: Statistics · Fall 2026**

> **Status:** 🚧 Experiment design finalized; data collection in progress. Results will be added once the runs are complete.

## Overview

A popular piece of "prompt engineering folklore" says that large language models give better answers if you offer them a tip. This project tests that claim with a controlled, paired experiment.

We ask the same small language model (served through [Tinker](https://thinkingmachines.ai/tinker/)) every question in the **GSM8K test set (1,319 grade-school math word problems)** under two prompt conditions:

| Condition | Text added to the prompt |
|---|---|
| **No tip** | "I'm going to tip $0 for a perfect answer." |
| **Tip** | "I'm going to tip $100 for a perfect answer." |

Everything else — the model, the question, the instructions, and the decoding settings — is held constant. Each answer is graded automatically as **correct** or **incorrect**, and we compare accuracy between the two conditions.

## Research Question & Hypotheses

**Research question:** Does offering a $100 tip in the prompt change a small LLM's accuracy on grade-school math problems, compared with offering a $0 tip?

- **H₀ (null):** The probability of a correct answer is the same under both prompts. Equivalently, among questions where the two prompts disagree, "correct only with tip" and "correct only without tip" are equally likely.
- **H₁ (alternative):** The probability of a correct answer differs between the two prompts (two-sided).

## Experimental Design

| Element | Choice |
|---|---|
| **Design** | Paired (within-question): every question is answered under both conditions |
| **Independent variable** | Prompt condition: $0 tip vs. $100 tip |
| **Dependent variable** | Whether the final answer is correct (binary) |
| **Unit of analysis** | One GSM8K test question (n = 1,319) |
| **Model** | Small open-weight model via Tinker — *exact model name TBD* |
| **Decoding** | Greedy (temperature = 0) so every run is deterministic and reproducible |
| **Controls** | Identical system/user template and answer-format instruction; the two tip sentences are matched in wording and length; same model and settings for both conditions |
| **Ordering** | Question order and condition order are randomized with a fixed seed |
| **Grading** | Model is asked to end with `Answer: <number>`; the number is extracted by script and compared with the GSM8K reference. Unparseable outputs are counted as incorrect (rule fixed *before* running). |

## Statistical Analysis Plan

- **Primary test:** **McNemar's test** on the 2×2 table of paired outcomes (exact binomial version if the number of discordant pairs is small).
- **Effect size:** Difference in accuracy (tip − no tip) with a 95% confidence interval, plus the paired odds ratio (b / c).
- **Power analysis:** See [`src/power_analysis.py`](src/power_analysis.py). Prior work suggests tipping effects are small, so we plan for a small difference; using the full test set (n = 1,319) is what makes the experiment adequately powered.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── load_data.py        # download GSM8K test set → data/raw/
│   ├── prompts.py          # the two prompt templates
│   ├── run_experiment.py   # query the model under both conditions → data/raw/
│   ├── grade.py            # extract answers, score correct/incorrect → data/processed/
│   ├── power_analysis.py   # sample-size / power calculation for McNemar's test
│   └── analyze.py          # McNemar's test, effect sizes, CIs, figures → results/
├── data/
│   ├── raw/                # GSM8K questions + raw model outputs
│   └── processed/          # graded, paired results
├── results/
│   └── figures/
└── report/                 # written report
```

## How to Reproduce

> Steps are filled in as scripts are completed.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Tinker API key
export TINKER_API_KEY=<your-key>

# 3. Run the pipeline
python src/power_analysis.py      # sample-size justification
python src/load_data.py           # download GSM8K test set
python src/run_experiment.py      # query the model (both conditions)
python src/grade.py               # score the answers
python src/analyze.py             # statistics + figures
```

## Results

*Coming soon.*

## Limitations (planned discussion)

- Results apply to one small model and one task (math word problems); they may not generalize to larger models or other tasks.
- Greedy decoding removes sampling noise but shows only one possible answer per prompt.
- "$0 tip" is itself a framing and may not be a fully neutral baseline.

## Team

- *Name 1*
- *Name 2*

## References

- Cobbe, K., et al. (2021). *Training Verifiers to Solve Math Word Problems.* arXiv:2110.14168. (GSM8K)
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153–157.
