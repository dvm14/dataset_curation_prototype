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

    specs_by_discipline = {
        spec["discipline"].lower(): spec for spec in speaker_specs
    }

    def to_enum(enum_cls, value, fallback):
        """
        Match the LLM's value to the enum ignoring case/whitespace
        (e.g. "Principal Investigator"); otherwise use the sampled value.
        """
        if isinstance(value, str):
            for member in enum_cls:
                if member.value.lower() == value.strip().lower():
                    return member
        return enum_cls(fallback)

    speakers = []
    for i, s in enumerate(raw.get("speakers", [])):
        spec = specs_by_discipline.get(
            str(s.get("discipline", "")).lower(),
            speaker_specs[i % len(speaker_specs)],
        )
        speakers.append(SpeakerProfile(
            name=s["name"],
            discipline=s["discipline"],
            functional_role=to_enum(FunctionalRole, s.get("functional_role"), spec["functional_role"]),
            background=s["background"],
            distinct_perspective=s["distinct_perspective"],
            tone=to_enum(Tone, s.get("tone"), spec["tone"]),
            language_complexity=to_enum(LanguageComplexity, s.get("language_complexity"), spec["language_complexity"]),
            communication_style=to_enum(CommunicationStyle, s.get("communication_style"), spec["communication_style"]),
            epistemic_stance=to_enum(EpistemicStance, s.get("epistemic_stance"), spec["epistemic_stance"]),
        ))

    return CastingOutput(speakers=speakers)
