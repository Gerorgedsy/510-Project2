"""Prompt template and the three experimental conditions.

Every condition uses the same instruction; only the problem text differs.
  original         the GSM8K test problem
  distractor       the same problem with GSM-Plus's inserted, topic-related but irrelevant sentence
  original_repeat  a word-for-word copy of original (A/A negative control: differences are pure noise)
"""

CONDITIONS = ("original", "distractor", "original_repeat")
QUESTION_COLUMN = {
    "original": "original_question",
    "distractor": "distractor_question",
    "original_repeat": "original_question",
}

TEMPLATE = (
    "Solve the following math word problem. Show your reasoning step by step, "
    "then write the final numeric answer on the last line in the form "
    "'Answer: <number>'.\n\n"
    "Question: {question}"
)


def build_prompt(row, condition: str) -> str:
    """Return the full user message for one question row under one condition."""
    if condition not in QUESTION_COLUMN:
        raise ValueError(f"Unknown condition: {condition}")
    return TEMPLATE.format(question=row[QUESTION_COLUMN[condition]])
