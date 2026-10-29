"""
agent1_content_curation.py
Content Curation Agent — takes the research setup and benchmark task,
returns a meeting scenario brief with the seed embedded.
"""

import sys
sys.path.append("..")

from schema import ContentCurationOutput, SeedSpec, DisciplineDistance
from utils.llm import call_llm_json
from prompts.agent1_prompt import SYSTEM_PROMPT

ASSUMPTION_FIELDS = {"assumption", "dependent_decision", "disciplinary_context", "why_needed", "risk_if_false"}


def run(
    research_problem: str,
    disciplines: list[str],
    discipline_distance: DisciplineDistance,
    benchmark_task: str,
    benchmark_category: str,
) -> ContentCurationOutput:
    """
    Input:
        research_problem     — what the team is working on
        disciplines          — list of fields in the meeting
        discipline_distance  — close / distant / mixed
        benchmark_task       — the evaluation task to embed as a seed
        benchmark_category   — which benchmark category (jargon, assumptions, etc.)

    Output:
        ContentCurationOutput with research context + embedded SeedSpec
    """

    user_message = f"""
Research problem: {research_problem}

Disciplines in the meeting: {', '.join(disciplines)}

Discipline distance: {discipline_distance.value}

Benchmark category: {benchmark_category}

Benchmark task to embed as a seed:
{benchmark_task}

{"For assumptions, seed exactly one premise that a concrete joint research decision relies on. Ground truth must contain assumption, dependent_decision, disciplinary_context, why_needed, risk_if_false. State risk conditionally; do not invent transcript turns or quotes. The premise need not be false. Describe method, broader research goal and decision as distinct elements." if benchmark_category == "assumptions" else ""}

Return a JSON object matching this exact schema:
{{
  "research_problem": "...",
  "disciplines": [...],
  "discipline_distance": "close|distant|mixed",
  "meeting_context": "...",
  "seed": {{
    "benchmark_category": "...",
    "benchmark_task": "...",
    "explicit_instruction": "...",
    "ground_truth": {{}}
  }}
}}
"""

    for attempt in range(3 if benchmark_category == "assumptions" else 1):
        raw = call_llm_json(SYSTEM_PROMPT, user_message)
        if benchmark_category != "assumptions":
            break
        truth = raw.get("seed", {}).get("ground_truth", {})
        valid = (isinstance(truth, dict) and
                 all(isinstance(truth.get(key), str) and truth[key].strip()
                     for key in ASSUMPTION_FIELDS) and
                 not any("turn" in key.lower() or "quote" in key.lower()
                         for key in truth))
        if valid:
            break
        if attempt < 2:
            user_message += "\nYour previous assumptions ground truth had missing, blank, or premature evidence fields. Regenerate all five required nonempty fields; omit turn numbers and quotes."

    # parse seed
    seed_data = raw.get("seed", {})
    seed = SeedSpec(
        benchmark_category=seed_data.get("benchmark_category", benchmark_category),
        benchmark_task=seed_data.get("benchmark_task", benchmark_task),
        explicit_instruction=seed_data.get("explicit_instruction", ""),
        ground_truth=seed_data.get("ground_truth", {}),
    )
    if benchmark_category == "assumptions":
        missing = ASSUMPTION_FIELDS - set(seed.ground_truth)
        if missing or any(not isinstance(seed.ground_truth[k], str) or not seed.ground_truth[k].strip()
                          for k in ASSUMPTION_FIELDS - missing):
            raise ValueError(f"Incomplete assumptions ground truth: {sorted(missing)}")
        if any("turn" in key.lower() or "quote" in key.lower() for key in seed.ground_truth):
            raise ValueError("Ground truth must not invent transcript evidence.")

    return ContentCurationOutput(
        research_problem=raw.get("research_problem", research_problem),
        disciplines=raw.get("disciplines", disciplines),
        discipline_distance=DisciplineDistance(
            raw.get("discipline_distance", discipline_distance.value)
        ),
        meeting_context=raw.get("meeting_context", ""),
        seed=seed,
    )
