"""Shared OpenAI Responses API calls for all generation and QA agents."""

import json
import os
import threading
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[1] / '.env')

# Lazy initialization allows offline imports/tests without credentials.
client = None
_client_lock = threading.Lock()
_usage = defaultdict(lambda: defaultdict(int))
_usage_lock = threading.Lock()


def _get_client():
    global client
    with _client_lock:
        if client is None:
            if not os.getenv('OPENAI_API_KEY', '').strip():
                raise ValueError('Set OPENAI_API_KEY in the repository .env before running.')
            client = OpenAI(timeout=120.0, max_retries=2)
        return client


def _record_usage(model, usage):
    if usage is None:
        return
    details = usage.input_tokens_details
    with _usage_lock:
        totals = _usage[model]
        totals['input'] += usage.input_tokens
        totals['cached'] += getattr(details, 'cached_tokens', 0) or 0
        totals['output'] += usage.output_tokens


def usage_report() -> str:
    """Cumulative session usage. Cached tokens are a subset of input tokens."""
    with _usage_lock:
        snapshot = {model: dict(totals) for model, totals in _usage.items()}
    lines = [
        f"  {model}: {t['input']:,} input ({t['cached']:,} cached), "
        f"{t['output']:,} output"
        for model, t in snapshot.items()
    ]
    return '\n'.join(lines + [
        '  Session totals; cached tokens are included in input. '
        'See OpenAI billing for cost.'
    ])


def _request(system_prompt, user_message, max_tokens, model, json_mode=False):
    model = (model or os.getenv('OPENAI_MODEL', '')).strip()
    if not model:
        raise ValueError('Set OPENAI_MODEL to a model available to your OpenAI project.')

    # Preserve block order and context boundaries, discarding provider metadata.
    if isinstance(user_message, str):
        content = user_message
    else:
        content = []
        for block in user_message:
            if block.get('type') not in ('text', 'input_text') or not isinstance(block.get('text'), str):
                raise ValueError('Only text content blocks are supported.')
            content.append({'type': 'input_text', 'text': block['text']})

    if json_mode:
        instruction = 'Return a valid JSON object.'
        if isinstance(content, str):
            content += '\n\n' + instruction
        else:
            content.append({'type': 'input_text', 'text': instruction})

    # Reasoning models may need a larger budget than the original short bids.
    # A configured floor applies to every request, including retries.
    minimum = int(os.getenv('OPENAI_MIN_OUTPUT_TOKENS', '0'))
    if minimum < 0 or max_tokens <= 0:
        raise ValueError('Output token budgets must be positive (minimum may be zero).')
    kwargs = dict(
        model=model, instructions=system_prompt,
        input=[{'role': 'user', 'content': content}],
        max_output_tokens=max(max_tokens, minimum), store=False,
    )
    effort = os.getenv('OPENAI_REASONING_EFFORT', '').strip()
    if effort:
        kwargs['reasoning'] = {'effort': effort}
    if json_mode:
        kwargs['text'] = {'format': {'type': 'json_object'}}

    response = _get_client().responses.create(**kwargs)
    _record_usage(response.model or model, response.usage)
    if response.status != 'completed':
        reason = getattr(response.incomplete_details, 'reason', None)
        raise ValueError(
            f'OpenAI response {response.status}: {reason or "no completion"}. '
            'For max_output_tokens, increase OPENAI_MIN_OUTPUT_TOKENS or use '
            'a supported lower OPENAI_REASONING_EFFORT.'
        )
    for item in response.output:
        for part in getattr(item, 'content', []):
            if part.type == 'refusal':
                raise ValueError('OpenAI refused the request; no transcript was produced.')
    if not response.output_text.strip():
        raise ValueError('OpenAI returned empty text.')
    return response.output_text


def call_llm(system_prompt: str, user_message: str | list[dict],
             max_tokens: int = 4000, model: str | None = None) -> str:
    """Free text, used by the AMI style extractor."""
    return _request(system_prompt, user_message, max_tokens, model)


def call_llm_json(system_prompt: str, user_message: str | list[dict],
                  max_tokens: int = 4000, model: str | None = None) -> dict:
    """JSON object mode; agent-level Pydantic models validate field schemas.

    Retry malformed/non-object JSON at most three times. API errors, refusals
    and incomplete responses propagate instead of being mistaken for valid data.
    """
    instructions = system_prompt + '\n\nReturn only a valid JSON object. No markdown or preamble.'
    for attempt in range(3):
        raw = _request(instructions, user_message, max_tokens, model, json_mode=True)
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError('Expected a JSON object.')
            return parsed
        except (json.JSONDecodeError, ValueError) as exc:
            if attempt == 2:
                raise ValueError('OpenAI returned invalid JSON objects after 3 attempts.') from exc
            print(f'[llm] Invalid JSON object (attempt {attempt + 1}), retrying...')
    raise AssertionError('Unreachable')
