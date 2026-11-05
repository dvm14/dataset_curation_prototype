"""
agent1_content_curation.py
Content Curation Agent — takes the research setup and benchmark task,
returns a meeting scenario brief with the seed embedded.
"""

import sys
sys.path.append("..")

from schema import ContentCurationOutput, SeedSpec, DisciplineDistance
from utils.llm import call_llm_json
from utils.seed_terms import is_term_task
from prompts.agent1_prompt import SYSTEM_PROMPT


# Keys Agent 1 has used for a scripted example meeting inside ground_truth.
# The simulator reads ground_truth, so a script there gets acted out.
DIALOGUE_KEYS = ("transcript", "dialogue", "utterance", "example_exchange", "script")


def strip_dialogue(ground_truth):
    if not isinstance(ground_truth, dict):
        return ground_truth
    dropped = [k for k in ground_truth if any(d in k.lower() for d in DIALOGUE_KEYS)]
    if dropped:
        print(f"      (dropped scripted dialogue from ground truth: {', '.join(dropped)})")
    return {k: v for k, v in ground_truth.items() if k not in dropped}


def run(
    research_problem: str,
    disciplines: list[str],
    discipline_distance: DisciplineDistance,
    benchmark_task: str,
    benchmark_category: str,
    task_id: str = "custom",
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

    # Terminology tasks are checked by exact match on the term
    # (utils/seed_terms.py), so it has to be under a predictable key.
    term_note = ""
    if is_term_task(benchmark_category, benchmark_task):
        term_note = (
            'Write ground_truth as {"term": "<the word as speakers say it>", '
            '"meanings": [{"discipline": "<one of the disciplines above>", '
            '"meaning": "..."}, ...]}, one entry per discipline that uses the '
            "term in its own sense.\n\n"
        )

    user_message = f"""
Research problem: {research_problem}

Disciplines in the meeting: {', '.join(disciplines)}

Discipline distance: {discipline_distance.value}

Task ID: {task_id}
Benchmark category: {benchmark_category}

Benchmark task to embed as a seed:
{benchmark_task}

{term_note}Return a JSON object matching this exact schema:
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

    raw = call_llm_json(SYSTEM_PROMPT, user_message)

    # parse seed
    seed_data = raw.get("seed", {})
    ground_truth = strip_dialogue(seed_data.get("ground_truth", {}))
    seed = SeedSpec(
        task_id=task_id,
        benchmark_category=benchmark_category,
        benchmark_task=benchmark_task,
        explicit_instruction=seed_data.get("explicit_instruction", ""),
        ground_truth=ground_truth,
    )

    return ContentCurationOutput(
        research_problem=raw.get("research_problem", research_problem),
        disciplines=raw.get("disciplines", disciplines),
        discipline_distance=DisciplineDistance(
            raw.get("discipline_distance", discipline_distance.value)
        ),
        meeting_context=raw.get("meeting_context", ""),
        seed=seed,
    )