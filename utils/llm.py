"""
llm.py
Shared LLM call utility used by all agents.
All agents use claude-sonnet-4-6 and return JSON.
"""

import anthropic
import json
import re
import os
import threading
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-6"

# $ per million tokens: (input, output). Cache writes cost 1.25x input, reads 0.1x.
PRICES = {
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# Token totals per model for the current run, so main.py can report actual cost.
_usage = defaultdict(lambda: defaultdict(int))
_usage_lock = threading.Lock()


def _record_usage(model, usage):
    with _usage_lock:
        totals = _usage[model]
        totals["input"] += usage.input_tokens
        totals["cache_write"] += usage.cache_creation_input_tokens or 0
        totals["cache_read"] += usage.cache_read_input_tokens or 0
        totals["output"] += usage.output_tokens


def usage_report() -> str:
    """Token totals and estimated cost for every call made so far."""
    lines, total_cost = [], 0.0
    for model, t in _usage.items():
        price_in, price_out = PRICES.get(model, (0, 0))
        cost = (
            t["input"] * price_in
            + t["cache_write"] * price_in * 1.25
            + t["cache_read"] * price_in * 0.1
            + t["output"] * price_out
        ) / 1e6
        total_cost += cost
        lines.append(
            f"  {model}: {t['input']:,} input, {t['cache_write']:,} cache write, "
            f"{t['cache_read']:,} cache read, {t['output']:,} output → ${cost:.2f}"
        )
    return "\n".join(lines + [f"  total ≈ ${total_cost:.2f}"])


def call_llm(
    system_prompt: str,
    user_message: str | list[dict],
    max_tokens: int = 4000,
    model: str = MODEL,
) -> str:
    """
    Single LLM call. Returns raw text response.
    Used by agents that return free text (e.g. style extraction).

    user_message can be a plain string, or a list of content blocks
    (e.g. with cache_control markers) for prompts that are reused.
    """
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": user_message}],
        system=system_prompt,
    )
    _record_usage(model, response.usage)
    return response.content[0].text


def call_llm_json(
    system_prompt: str,
    user_message: str | list[dict],
    max_tokens: int = 4000,
    model: str = MODEL,
) -> dict:
    """
    LLM call that returns parsed JSON.
    Used by agents that return structured output.
    Retries up to 3 times on parse failure.
    """
    system_with_json = (
        system_prompt
        + "\n\nIMPORTANT: Return ONLY valid JSON. No preamble, no markdown, "
          "no code fences. Your entire response must be parseable as JSON."
    )

    for attempt in range(3):
        raw = call_llm(system_with_json, user_message, max_tokens, model)
        try:
            # strip any accidental markdown fences
            cleaned = re.sub(r"```json|```", "", raw).strip()
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            if attempt == 2:
                raise ValueError(
                    f"LLM returned invalid JSON after 3 attempts.\n"
                    f"Last error: {e}\nLast response: {raw[:500]}"
                )
            print(f"[llm] JSON parse failed (attempt {attempt+1}), retrying...")

    return {}
