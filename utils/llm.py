"""
llm.py
Shared LLM call utility used by all agents.
All agents use claude-sonnet-4-6 and return JSON.
"""

import anthropic
import json
import re
import os
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-6"


def call_llm(system_prompt: str, user_message: str, max_tokens: int = 4000) -> str:
    """
    Single LLM call. Returns raw text response.
    Used by agents that return free text (e.g. style extraction).
    """
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": user_message}],
        system=system_prompt,
    )
    return response.content[0].text


def call_llm_json(system_prompt: str, user_message: str, max_tokens: int = 4000) -> dict:
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
        raw = call_llm(system_with_json, user_message, max_tokens)
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
