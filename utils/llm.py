"""
llm.py
Shared LLM call utility used by all agents. Every call returns text or JSON.

Two roles, set in .env (both default to gpt-5.6-terra):
    GEN_MODEL    writes the meeting: Agents 1-3, style notes, series planner/summaries
    JUDGE_MODEL  grades it: Agent 4 QA and series QA

A model whose name starts with "claude-" goes to the Anthropic API
(ANTHROPIC_API_KEY); anything else goes to OpenAI (OPENAI_API_KEY). Keeping the
judge fixed while the generator changes means transcripts from different
generators are graded the same way.
"""

import json
import os
import re
import threading
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()

GEN_MODEL = os.getenv("GEN_MODEL", "gpt-5.6-terra")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-5.6-terra")
MODEL = GEN_MODEL  # kept for older imports

# OpenAI: no hidden reasoning tokens, so max_tokens limits only the visible reply.
REASONING_EFFORT = "none"

# Claude models think adaptively and that cannot be switched off on current
# models; "low" keeps thinking short for these small, frequent calls.
CLAUDE_EFFORT = os.getenv("CLAUDE_EFFORT", "low")
# Thinking counts toward max_tokens on Claude, so give it room on top of the
# visible reply budget each call asks for.
CLAUDE_THINKING_HEADROOM = 4000

# Caching only happens at blocks marked with this (see agent3's participant_blocks).
CACHE_BREAKPOINT = {"mode": "explicit"}

# $ per million tokens: (input, cached input, output). Cache writes are priced
# at 1.25x input. Check openai.com/api/pricing and platform.claude.com pricing.
PRICES = {
    "gpt-5.6-terra": (2.00, 0.20, 12.00),
    "claude-sonnet-5-5": (2.00, 0.20, 10.00),
    "claude-opus-5-5": (4.00, 0.40, 20.00),
}

def output_root():
    """
    data/outputs/<generator>/ for runs that set GEN_MODEL, so Claude and GPT
    samples never mix; data/outputs/ when it isn't set, as before.
    RUN_TAG=x adds a suffix (data/outputs/<generator>__x/) to keep a rerun
    on changed code apart from earlier samples.
    """
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "data" / "outputs"
    if not os.getenv("GEN_MODEL"):
        return root
    tag = os.getenv("RUN_TAG", "").strip()
    return root / (f"{GEN_MODEL}__{tag}" if tag else GEN_MODEL)


_clients = {}
_clients_lock = threading.Lock()


def is_claude(model: str) -> bool:
    return model.startswith("claude-")


def _client(provider: str):
    with _clients_lock:
        if provider not in _clients:
            if provider == "anthropic":
                from anthropic import Anthropic
                # Extra retries: 429/529 are common in long batch runs.
                _clients[provider] = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), max_retries=6)
            else:
                from openai import OpenAI
                _clients[provider] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return _clients[provider]


# ── usage and cost ────────────────────────────────────────────────
_usage = defaultdict(lambda: defaultdict(int))
_usage_lock = threading.Lock()


def _add_usage(model, fresh_input, cache_write, cache_read, output):
    with _usage_lock:
        totals = _usage[model]
        totals["input"] += fresh_input
        totals["cache_write"] += cache_write
        totals["cache_read"] += cache_read
        totals["output"] += output


def _record_openai_usage(model, usage):
    details = usage.prompt_tokens_details
    cached = getattr(details, "cached_tokens", 0) or 0
    written = (getattr(details, "model_extra", None) or {}).get("cache_write_tokens", 0) or 0
    _add_usage(model, usage.prompt_tokens - cached - written, written, cached, usage.completion_tokens)


def _record_claude_usage(model, usage):
    _add_usage(
        model,
        usage.input_tokens or 0,
        getattr(usage, "cache_creation_input_tokens", 0) or 0,
        getattr(usage, "cache_read_input_tokens", 0) or 0,
        usage.output_tokens or 0,  # includes thinking tokens, which are billed as output
    )


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
        note = "" if model in PRICES else " (no price on file; cost not counted)"
        lines.append(
            f"  {model}: {t['input']:,} input, {t['cache_write']:,} cache write, "
            f"{t['cache_read']:,} cache read, "
            f"{t['output']:,} output → ${cost:.2f}{note}"
        )
    return "\n".join(lines + [f"  total ≈ ${total_cost:.2f}"])


# ── calls ─────────────────────────────────────────────────────────
def _claude_blocks(user_message):
    """
    Convert the shared message format to Anthropic content blocks: a block
    marked with prompt_cache_breakpoint gets cache_control, at most 4 per
    request (the API limit; the last ones are kept).
    """
    if isinstance(user_message, str):
        return user_message
    blocks = [{"type": "text", "text": b["text"]} for b in user_message]
    marked = [i for i, b in enumerate(user_message) if b.get("prompt_cache_breakpoint")]
    for i in marked[-4:]:
        blocks[i]["cache_control"] = {"type": "ephemeral"}
    return blocks


def _call_claude(system_prompt, user_message, max_tokens, model):
    response = _client("anthropic").messages.create(
        model=model,
        max_tokens=max_tokens + CLAUDE_THINKING_HEADROOM,
        system=system_prompt,
        messages=[{"role": "user", "content": _claude_blocks(user_message)}],
        thinking={"type": "adaptive"},
        output_config={"effort": CLAUDE_EFFORT},
    )
    _record_claude_usage(model, response.usage)
    text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
    if response.stop_reason == "max_tokens":
        print(f"[llm] {model} hit max_tokens; reply may be cut off")
    return text


def _call_openai(system_prompt, user_message, max_tokens, model, json_mode):
    response = _client("openai").chat.completions.create(
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
    _record_openai_usage(model, response.usage)
    return response.choices[0].message.content or ""


def call_llm(
    system_prompt: str,
    user_message: str | list[dict],
    max_tokens: int = 4000,
    model: str | None = None,
    json_mode: bool = False,
) -> str:
    """
    Single LLM call. Returns raw text. model defaults to GEN_MODEL.

    user_message can be a plain string, or a list of content blocks
    (e.g. with prompt_cache_breakpoint markers) for prompts that are reused.
    """
    model = model or GEN_MODEL
    if is_claude(model):
        return _call_claude(system_prompt, user_message, max_tokens, model)
    return _call_openai(system_prompt, user_message, max_tokens, model, json_mode)


def _parse_json(raw: str):
    cleaned = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Claude has no generic JSON mode; take the outermost object if there
        # is text around it.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise


def call_llm_json(
    system_prompt: str,
    user_message: str | list[dict],
    max_tokens: int = 4000,
    model: str | None = None,
) -> dict:
    """
    LLM call that returns parsed JSON. model defaults to GEN_MODEL; QA code
    passes JUDGE_MODEL. Retries up to 3 times on parse failure.
    """
    system_with_json = (
        system_prompt
        + "\n\nIMPORTANT: Return ONLY valid JSON. No preamble, no markdown, "
          "no code fences. Your entire response must be parseable as JSON."
    )

    for attempt in range(3):
        raw = call_llm(system_with_json, user_message, max_tokens, model, json_mode=True)
        try:
            return _parse_json(raw)
        except json.JSONDecodeError as e:
            if attempt == 2:
                raise ValueError(
                    f"LLM returned invalid JSON after 3 attempts.\n"
                    f"Last error: {e}\nLast response: {raw[:500]}"
                )
            print(f"[llm] JSON parse failed (attempt {attempt+1}), retrying...")

    return {}
