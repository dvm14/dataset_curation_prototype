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

    raw = call_llm_json(SYSTEM_PROMPT, user_message)

    # parse seed
    seed_data = raw.get("seed", {})
    seed = SeedSpec(
        benchmark_category=seed_data.get("benchmark_category", benchmark_category),
        benchmark_task=seed_data.get("benchmark_task", benchmark_task),
        explicit_instruction=seed_data.get("explicit_instruction", ""),
        ground_truth=seed_data.get("ground_truth", {}),
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
