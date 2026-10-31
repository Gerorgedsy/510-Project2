"""Thin wrapper around the Tinker sampling client for single-turn chat prompts."""

import os

import tinker
from dotenv import load_dotenv
from tinker import types

from config import ENABLE_THINKING, MAX_TOKENS, MODEL_NAME, REPO_ENV, SAMPLING_SEED, TEMPERATURE


async def make_sampler():
    """Return (sampling_client, tokenizer) for MODEL_NAME; the API key comes from the repo-root .env."""
    load_dotenv(REPO_ENV)
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit(f"TINKER_API_KEY not found; put it in {REPO_ENV}")
    service = tinker.ServiceClient()
    client = await service.create_sampling_client_async(base_model=MODEL_NAME)
    return client, client.get_tokenizer()


async def generate(client, tokenizer, prompt: str) -> dict:
    """Send one user message (wrapped in the chat template) and return the answer plus metadata.

    stop_reason == "length" means the answer hit MAX_TOKENS and was cut off.
    n_output_tokens counts generated tokens, including the end-of-turn token when present.
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )
    prompt_ids = tokenizer.encode(text, add_special_tokens=False)
    result = await client.sample_async(
        prompt=types.ModelInput.from_ints(prompt_ids),
        num_samples=1,
        sampling_params=types.SamplingParams(max_tokens=MAX_TOKENS, temperature=TEMPERATURE, seed=SAMPLING_SEED),
    )
    seq = result.sequences[0]
    return {
        "response": tokenizer.decode(seq.tokens, skip_special_tokens=True),
        "stop_reason": seq.stop_reason,
        "n_prompt_tokens": len(prompt_ids),
        "n_output_tokens": len(seq.tokens),
    }
