"""
llm.py
Shared LLM call utility used by all agents.
All agents use gpt-5.6-terra and return JSON.
"""

import json
import re
import os
import threading
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-5.6-terra"

# No hidden reasoning tokens, so max_tokens limits only the visible reply.
REASONING_EFFORT = "none"

# Caching only happens at blocks marked with this (see agent3's participant_blocks).
CACHE_BREAKPOINT = {"mode": "explicit"}

# $ per million tokens: (input, cached input, output). Cache writes cost 1.25x input.
# Cached input rate is an estimate; check openai.com/api/pricing.
PRICES = {
    "gpt-5.6-terra": (2.00, 0.20, 12.00),
}

# Token totals per model for the current run, so main.py can report actual cost.
_usage = defaultdict(lambda: defaultdict(int))
_usage_lock = threading.Lock()


def _record_usage(model, usage):
    details = usage.prompt_tokens_details
    cached = getattr(details, "cached_tokens", 0) or 0
    written = (getattr(details, "model_extra", None) or {}).get("cache_write_tokens", 0) or 0
    with _usage_lock:
        totals = _usage[model]
        totals["input"] += usage.prompt_tokens - cached - written
        totals["cache_write"] += written
        totals["cache_read"] += cached
        totals["output"] += usage.completion_tokens


def usage_report() -> str:
    """Token totals and estimated cost for every call made so far."""
    lines, total_cost = [], 0.0
    for model, t in _usage.items():
        price_in, price_cached, price_out = PRICES.get(model, (0, 0, 0))
        cost = (
            t["input"] * price_in
            + t["cache_write"] * price_in * 1.25
            + t["cache_read"] * price_cached
            + t["output"] * price_out
        ) / 1e6
        total_cost += cost
        lines.append(
            f"  {model}: {t['input']:,} input, {t['cache_write']:,} cache write, "
            f"{t['cache_read']:,} cache read, "
            f"{t['output']:,} output → ${cost:.2f}"
        )
    return "\n".join(lines + [f"  total ≈ ${total_cost:.2f}"])


def call_llm(
    system_prompt: str,
    user_message: str | list[dict],
    max_tokens: int = 4000,
    model: str = MODEL,
    json_mode: bool = False,
) -> str:
    """
    Single LLM call. Returns raw text response.
    Used by agents that return free text (e.g. style extraction).

    user_message can be a plain string, or a list of content blocks
    (e.g. with prompt_cache_breakpoint markers) for prompts that are reused.
    """
    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens,
        reasoning_effort=REASONING_EFFORT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        extra_body={"prompt_cache_options": {"mode": "explicit"}},
        **({"response_format": {"type": "json_object"}} if json_mode else {}),
    )
    _record_usage(model, response.usage)
    return response.choices[0].message.content or ""


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
        raw = call_llm(system_with_json, user_message, max_tokens, model, json_mode=True)
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
