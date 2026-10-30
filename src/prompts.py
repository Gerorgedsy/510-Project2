"""Prompt templates for the two experimental conditions.

The two tip sentences are identical except for the dollar amount, so the
only thing that changes between conditions is the size of the tip.
"""

CONDITIONS = {
    "tip_0": "I'm going to tip $0 for a perfect answer.",
    "tip_100": "I'm going to tip $100 for a perfect answer.",
}

TEMPLATE = (
    "Solve the following math word problem. Show your reasoning step by step, "
    "then write the final numeric answer on the last line in the form "
    "'Answer: <number>'.\n\n"
    "{tip_sentence}\n\n"
    "Question: {question}"
)


def build_prompt(question: str, condition: str) -> str:
    """Return the full prompt for one question under one condition."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    return TEMPLATE.format(tip_sentence=CONDITIONS[condition], question=question)
