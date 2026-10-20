"""
agent3_filming.py
Filming Agent — generates the full meeting transcript using
the scenario brief, speaker profiles, AMI style notes, and seed.
"""

import sys
sys.path.append("..")

from schema import (
    FilmingOutput, Turn,
    ContentCurationOutput, CastingOutput
)
from utils.llm import call_llm_json
from prompts.agent3_prompt import SYSTEM_PROMPT


def run(
    content: ContentCurationOutput,
    casting: CastingOutput,
    ami_style_notes: str,
    qa_feedback: str = None,
) -> FilmingOutput:
    """
    Input:
        content          — output of Agent 1 (scenario + seed)
        casting          — output of Agent 2 (speaker profiles)
        ami_style_notes  — extracted style observations from AMI transcript
        qa_feedback      — if this is a re-film, the QA agent's feedback

    Output:
        FilmingOutput with transcript turns + seed location note
    """

    # format speaker profiles for the prompt
    speaker_summary = "\n".join([
        f"- {s.name} ({s.discipline}): "
        f"{s.functional_role.value}, "
        f"{s.tone.value} tone, "
        f"{s.communication_style.value}, "
        f"{s.epistemic_stance.value}. "
        f"Perspective: {s.distinct_perspective}"
        for s in casting.speakers
    ])

    refilm_section = ""
    if qa_feedback:
        refilm_section = f"""
This is a RE-FILM. The previous version had issues. Fix the following:
{qa_feedback}
"""

    user_message = f"""
Research problem: {content.research_problem}

Meeting context: {content.meeting_context}

Speakers:
{speaker_summary}

SEED — embed this benchmark feature naturally in the transcript:
{content.seed.explicit_instruction}

AMI style reference — match this meeting's conversational style:
{ami_style_notes}

{refilm_section}

Generate a ~10 minute meeting transcript (20-30 turns).
Note where you embedded the seed.

Return JSON:
{{
  "transcript": [
    {{"speaker": "name", "text": "what they said"}},
    ...
  ],
  "seed_location": "brief description of where/how the seed appears"
}}
"""

    raw = call_llm_json(SYSTEM_PROMPT, user_message, max_tokens=6000)

    turns = [
        Turn(speaker=t["speaker"], text=t["text"])
        for t in raw.get("transcript", [])
    ]

    return FilmingOutput(
        transcript=turns,
        seed_location=raw.get("seed_location", "not specified"),
    )
