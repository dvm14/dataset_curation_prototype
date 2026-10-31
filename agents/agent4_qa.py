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
from agents.agent3_filming import seed_terms, term_uses, seed_speakers, count_mentions
from prompts.agent4_prompt import SYSTEM_PROMPT, SEED_USE_PROMPT, SPOKEN_CHECK_PROMPT


# Minimum rubric score for approval. The seed criteria are stricter:
# a 3 there ("needs inference" / "partly surfaced") isn't usable as a benchmark item.
APPROVAL_THRESHOLDS = {
    "seed_recoverable": 4,
    "seed_implicit": 4,
    "naturalness": 3,
    "speaker_distinctness": 3,
    "domain_alignment": 3,
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


# Turns shorter than this (reactions, quick answers) aren't labeled.
SPOKEN_MIN_WORDS = 15


def spoken_check(transcript):
    """
    The 1-5 naturalness score hardly moves between versions of a
    transcript, so label each substantive turn spoken or written and
    report the share. Returns (share, written turn numbers).
    """
    numbers = [
        i for i, t in enumerate(transcript, start=1)
        if len(t.text.split()) >= SPOKEN_MIN_WORDS
    ]
    if not numbers:
        return None, []

    turns_text = "\n".join(
        f"[{i}] {transcript[i - 1].speaker}: {transcript[i - 1].text}"
        for i in numbers
    )
    raw = call_llm_json(SPOKEN_CHECK_PROMPT, turns_text, max_tokens=2000)

    labels = {
        item.get("turn"): item.get("label")
        for item in raw.get("labels", []) if isinstance(item, dict)
    }
    written = [i for i in numbers if labels.get(i) != "spoken"]
    return round(1 - len(written) / len(numbers), 2), written


# Highest seed_implicit allowed when a seed turn comments on or renames the
# term, or a seed speaker never uses it plainly. The judge gave 5 to a
# transcript whose only anthropology use objected to the label.
SEED_USE_CAP = 3


def seed_use_check(transcript, terms, content, casting):
    """
    Label each turn with the seed term (and the turn after it, where a
    rename would show up) in a separate small call: inside the main judge
    call these labels came back all clean. The rules are applied here in
    code. Returns (labels, issue strings).
    """
    if not terms:
        return [], []
    numbers = sorted({
        n for i, t in enumerate(transcript, start=1)
        if count_mentions(t.text, terms)
        for n in (i, i + 1) if n <= len(transcript)
    })
    if not numbers:
        return [], []

    disciplines = {s.name: s.discipline for s in casting.speakers}
    truth = content.seed.ground_truth
    senses = truth.get("discipline_specific_meanings", truth)
    turns_text = "\n".join(
        f"[{n}] {transcript[n - 1].speaker} "
        f"({disciplines.get(transcript[n - 1].speaker, '')}): {transcript[n - 1].text}"
        for n in numbers
    )
    raw = call_llm_json(
        SEED_USE_PROMPT,
        f"SEED TERM: {terms}\n\nSENSES:\n{json.dumps(senses, indent=2)}\n\n"
        f"TURNS:\n{turns_text}",
        max_tokens=2000,
    )
    labels = [u for u in raw.get("turns", []) if isinstance(u, dict)]

    problems, plain_senses = [], []
    for use in labels:
        turn = use.get("turn")
        if use.get("comments_on_term") is True:
            problems.append(f"Turn {turn}: comments on the seed term instead of using it ({use.get('why', '')}).")
        elif use.get("renames") is True:
            problems.append(f"Turn {turn}: renames a sense of the seed term ({use.get('why', '')}).")
        elif turn in numbers and count_mentions(transcript[turn - 1].text, terms):
            plain_senses.append(str(use.get("sense", "")).lower().replace("_", " "))

    # Every seed speaker's sense needs at least one plain use, by anyone.
    # Same rough stem as seed_speakers, so "anthropologist" matches "anthropology".
    for speaker in seed_speakers(content, casting.speakers):
        d = speaker.discipline.lower().strip()
        if not any(d[:max(5, len(d) - 3)] in sense for sense in plain_senses):
            problems.append(
                f"The {speaker.discipline} sense of the seed term never appears "
                f"in a plain use of it."
            )
    return labels, problems


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

    # Exact counts, not another agent's reading: the judge once repeated
    # the seed locator's false claim that a speaker never said the term.
    terms = seed_terms(content.seed.ground_truth)
    measured_uses = (
        term_uses(filming.transcript, terms) if terms
        else "No single seed term to count."
    )

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

SEED TERM USES (measured by exact text match; treat as correct):
{measured_uses}

AMI style reference (what a natural meeting sounds like):
{ami_style_notes}

Score each criterion 1-5 and list specific issues.

Return JSON:
{{
  "seed_recoverable": 1-5,
  "seed_implicit": 1-5,
  "naturalness": 1-5,
  "speaker_distinctness": 1-5,
  "domain_alignment": 1-5,
  "dialogue_flow": 1-5,
  "evidence_turns": [turn numbers],
  "realism_signals": ["signal names found"],
  "issues": ["issue 1", "issue 2"]
}}
"""

    raw = call_llm_json(SYSTEM_PROMPT, user_message)

    scores = {key: score(raw, key) for key in APPROVAL_THRESHOLDS}
    issues = raw.get("issues", [])

    seed_uses, problems = seed_use_check(filming.transcript, terms, content, casting)
    if problems:
        scores["seed_implicit"] = min(scores["seed_implicit"], SEED_USE_CAP)
        issues = issues + problems

    approved = all(
        scores[key] >= minimum
        for key, minimum in APPROVAL_THRESHOLDS.items()
    )

    spoken_share, written_turns = spoken_check(filming.transcript)

    return QAFeedback(
        **scores,
        spoken_share=spoken_share,
        written_turns=written_turns,
        evidence_turns=[
            t for t in raw.get("evidence_turns", []) if isinstance(t, int)
        ],
        realism_signals=raw.get("realism_signals", []),
        seed_uses=seed_uses,
        issues=issues,
        approved=approved,
    )
