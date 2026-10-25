"""
agent4_qa.py
QA Agent — LLM-as-judge that scores the transcript.
The re-film loop lives in main.py.
"""

import sys
sys.path.append("..")

import json
import re
import statistics

from schema import QAFeedback, FilmingOutput, ContentCurationOutput, CastingOutput
from utils.llm import call_llm_json
from prompts.agent4_prompt import SYSTEM_PROMPT


# Minimum rubric score for approval. The seed criteria are stricter:
# a 3 there ("needs inference" / "partly surfaced") isn't usable as a benchmark item.
APPROVAL_THRESHOLDS = {
    "seed_recoverable": 4,
    "seed_implicit": 4,
    "naturalness": 4,   # most important check, so stricter
    "speaker_distinctness": 3,
    "topical_alignment": 3,
    "dialogue_flow": 3,
}


DISFLUENCY = re.compile(
    r"\b(um+|uh+|er|hmm+|like|you know|i mean|sorry|wait)\b|—|-\s|\.\.\.",
    re.IGNORECASE,
)


def turn_statistics(transcript):
    """Measured facts about turn shape, so the judge doesn't rely on impressions."""
    lengths = [len(t.text.split()) for t in transcript]
    back_to_back = sum(
        1 for a, b in zip(transcript, transcript[1:]) if a.speaker == b.speaker
    )
    markers = sum(len(DISFLUENCY.findall(t.text)) for t in transcript)
    turns_per_speaker = {}
    for t in transcript:
        turns_per_speaker[t.speaker] = turns_per_speaker.get(t.speaker, 0) + 1
    return (
        f"- turns: {len(lengths)}; median words per turn: "
        f"{statistics.median(lengths):.0f}; longest: {max(lengths)}\n"
        f"- turns under 20 words: {sum(n < 20 for n in lengths)} of {len(lengths)}\n"
        f"- turns over 120 words: {sum(n > 120 for n in lengths)}\n"
        f"- disfluency markers (fillers, dashes, 'sorry', 'wait'...): "
        f"{markers / max(sum(lengths), 1) * 100:.1f} per 100 words\n"
        f"- back-to-back turns by the same speaker: {back_to_back}\n"
        f"- turns per speaker: {turns_per_speaker}\n"
        "These are context, not targets: there is no required turn length."
    )


def score(raw, key):
    """Missing or malformed scores count as failing, not as a pass."""
    try:
        return max(1, min(5, int(raw.get(key, 1))))
    except (TypeError, ValueError):
        return 1


def run(
    filming: FilmingOutput,
    content: ContentCurationOutput,
    casting: CastingOutput,
    ami_style_notes: str,
) -> QAFeedback:
    """
    Input:
        filming         — output of Agent 3 to evaluate
        content         — output of Agent 1 (for seed reference)
        casting         — output of Agent 2 (for speaker reference)
        ami_style_notes — for naturalness comparison

    Output:
        QAFeedback with rubric scores, issues, and approval
    """

    # format transcript for judge, numbered so evidence turns can be cited
    transcript_text = "\n".join([
        f"[{i + 1}] {t.speaker}: {t.text}"
        for i, t in enumerate(filming.transcript)
    ])

    speaker_profiles = "\n".join(
        speaker.model_dump_json() for speaker in casting.speakers
    )

    user_message = f"""
Evaluate this interdisciplinary research meeting transcript.

TRANSCRIPT (turn numbers in brackets):
{transcript_text}

SPEAKER PERSONAS:
{speaker_profiles}

SEED that should be embedded:
{content.seed.explicit_instruction}

GROUND TRUTH the AI assistant should be able to derive:
{json.dumps(content.seed.ground_truth, indent=2)}

TURN STATISTICS (measured):
{turn_statistics(filming.transcript)}

SEED location reported by Filming agent:
{filming.seed_location}

AMI style reference (what a natural meeting sounds like):
{ami_style_notes}

Score each criterion 1-5 and list specific issues.

Return JSON:
{{
  "seed_recoverable": 1-5,
  "seed_implicit": 1-5,
  "naturalness": 1-5,
  "speaker_distinctness": 1-5,
  "topical_alignment": 1-5,
  "dialogue_flow": 1-5,
  "evidence_turns": [turn numbers],
  "realism_signals": ["signal names found"],
  "issues": ["issue 1", "issue 2"]
}}
"""

    raw = call_llm_json(SYSTEM_PROMPT, user_message)

    scores = {key: score(raw, key) for key in APPROVAL_THRESHOLDS}
    approved = all(
        scores[key] >= minimum
        for key, minimum in APPROVAL_THRESHOLDS.items()
    )

    return QAFeedback(
        **scores,
        evidence_turns=[
            t for t in raw.get("evidence_turns", []) if isinstance(t, int)
        ],
        realism_signals=raw.get("realism_signals", []),
        issues=raw.get("issues", []),
        approved=approved,
    )
