"""
agent2_casting.py
Casting Agent — generates speaker profiles for each discipline,
drawing from the value bank unless attributes are specified.
"""

import sys
sys.path.append("..")

from schema import (
    CastingOutput, SpeakerProfile, ContentCurationOutput,
    FunctionalRole, Tone, LanguageComplexity,
    CommunicationStyle, EpistemicStance
)
from utils.llm import call_llm_json
from utils.value_banks import sample_speaker_attributes
from prompts.agent2_prompt import SYSTEM_PROMPT


def run(
    content: ContentCurationOutput,
    speaker_overrides: dict[str, dict] = {},
) -> CastingOutput:
    """
    Input:
        content           — output of Agent 1
        speaker_overrides — optional dict keyed by discipline name,
                            with attribute overrides for that speaker.
                            e.g. {"ecology": {"tone": "formal"}}

    Output:
        CastingOutput with one SpeakerProfile per discipline
    """

    # build the attribute spec for each discipline
    # randomize unless overrides are provided
    speaker_specs = []
    for discipline in content.disciplines:
        overrides = speaker_overrides.get(discipline, {})
        attrs = sample_speaker_attributes(overrides)
        speaker_specs.append({
            "discipline": discipline,
            **{k: v.value if hasattr(v, "value") else v
               for k, v in attrs.items()}
        })

    user_message = f"""
Research problem: {content.research_problem}

Meeting context: {content.meeting_context}

Seed to be aware of: {content.seed.explicit_instruction}

Generate one speaker profile per discipline using these randomized attributes as a starting point.
You may adjust background and distinct_perspective to fit the research problem,
but keep the assigned tone, language_complexity, communication_style, and epistemic_stance.

Speaker attribute specs:
{speaker_specs}

Return a JSON object:
{{
  "speakers": [
    {{
      "name": "...",
      "discipline": "...",
      "functional_role": "...",
      "background": "...",
      "distinct_perspective": "...",
      "tone": "...",
      "language_complexity": "...",
      "communication_style": "...",
      "epistemic_stance": "..."
    }}
  ]
}}
"""

    raw = call_llm_json(SYSTEM_PROMPT, user_message)

    speakers = []
    for s in raw.get("speakers", []):
        speakers.append(SpeakerProfile(
            name=s["name"],
            discipline=s["discipline"],
            functional_role=FunctionalRole(s["functional_role"]),
            background=s["background"],
            distinct_perspective=s["distinct_perspective"],
            tone=Tone(s["tone"]),
            language_complexity=LanguageComplexity(s["language_complexity"]),
            communication_style=CommunicationStyle(s["communication_style"]),
            epistemic_stance=EpistemicStance(s["epistemic_stance"]),
        ))

    return CastingOutput(speakers=speakers)
