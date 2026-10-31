"""Prompt template and the four experimental conditions.

Every condition uses the same instruction and question; only the sentence appended
at the end differs. tip_0 and tip_100 are identical except for the dollar amount,
so comparing them isolates the money. tip_0_repeat is a word-for-word copy of tip_0
(A/A negative control): any difference between the two is pure run-to-run noise.
"""

CONDITIONS = {
    "none": None,
    "tip_0": "I'm going to tip $0 for a perfect answer!",
    "tip_100": "I'm going to tip $100 for a perfect answer!",
    "tip_0_repeat": "I'm going to tip $0 for a perfect answer!",
}

TEMPLATE = (
    "Solve the following math word problem. Show your reasoning step by step, "
    "then write the final numeric answer on the last line in the form "
    "'Answer: <number>'.\n\n"
    "Question: {question}"
)


def build_prompt(question: str, condition: str) -> str:
    """Return the full user message for one question under one condition."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    prompt = TEMPLATE.format(question=question)
    sentence = CONDITIONS[condition]
    return prompt if sentence is None else f"{prompt}\n\n{sentence}"
