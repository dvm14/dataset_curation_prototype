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


def pick_one_per_discipline(cast, disciplines):
    """
    The LLM sometimes casts extra people for roles named in the seed
    ("project lead", "education lead"): 4 or 6 speakers for 3 disciplines,
    which used to abort the run. Keep the first speaker for each discipline,
    fill any discipline it skipped from the leftovers, and drop the rest.
    """
    cast = [s for s in cast if isinstance(s, dict) and s.get("name")]
    chosen, used = [], set()
    for discipline in disciplines:
        for i, s in enumerate(cast):
            if i not in used and str(s.get("discipline", "")).strip().lower() == discipline.lower():
                chosen.append(s); used.add(i); break
        else:
            chosen.append(None)
    leftovers = [s for i, s in enumerate(cast) if i not in used]
    for slot, discipline in enumerate(disciplines):
        if chosen[slot] is None and leftovers:
            chosen[slot] = {**leftovers.pop(0), "discipline": discipline}
    if len(cast) != len(disciplines):
        print(f"      (casting returned {len(cast)} speakers for "
              f"{len(disciplines)} disciplines; kept one per discipline)")
    return [s for s in chosen if s is not None]


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

Benchmark task: {content.seed.benchmark_task}

Seed to be aware of: {content.seed.explicit_instruction}

Generate exactly {len(speaker_specs)} speaker profiles, one per discipline listed below,
using these randomized attributes as a starting point. Roles named in the seed
(e.g. "project lead") are filled by these speakers; don't add extra people.
You may adjust background and distinct_perspective to fit the research problem,
and adjust sampled attributes if necessary to satisfy the task (e.g. a dominant
and a quiet participant). Do not reveal hidden test answers in speaker profiles.

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
    for i, s in enumerate(pick_one_per_discipline(raw.get("speakers", []), content.disciplines)):
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