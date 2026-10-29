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


def validate_assumption_evidence(transcript, evidence):
    """Check provenance only; the judge and human reviewer assess semantics."""
    issues = []
    roles = set()
    if not isinstance(evidence, list):
        return ["Evidence must be a list of turn, quote, and role records."]
    for item in evidence:
        if not isinstance(item, dict):
            issues.append("Evidence record is not an object.")
            continue
        turn = item.get("turn")
        quote = item.get("quote")
        role = item.get("role")
        if type(turn) is not int or not 1 <= turn <= len(transcript):
            issues.append(f"Invalid evidence turn: {turn}.")
        elif not isinstance(quote, str) or not quote.strip() or quote not in transcript[turn - 1].text:
            issues.append(f"Evidence quote does not occur in turn {turn}.")
        if role in {"method", "goal", "decision"}:
            roles.add(role)
    for role in {"method", "goal", "decision"} - roles:
        issues.append(f"Missing {role} evidence.")
    return issues


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

    is_assumption = content.seed.benchmark_category == "assumptions"
    if is_assumption:
        user_message += """
For assumptions, judge the FINAL transcript only. The target premise may be
inferred but method, research goal and an actual shared decision relying on
that method must each appear in the dialogue. Do not fill gaps from seed,
personas, private briefs or outside knowledge. A generic discussion without
the dependent decision is missing. Mark exposed if a participant states or
challenges the target premise; resolved if the decision changes to remove its
dependence. Ordinary questions and unrelated disagreement are neither.
Do not assert the assumption is false. Return additional JSON fields:
"assumption_status": "implicit"|"exposed"|"resolved"|"missing",
"evidence": [{"turn": 1, "quote": "exact substring of that turn", "role": "method"|"goal"|"decision"}],
"inference": "how the cited dialogue supports the unstated premise".
"implicit" requires all three evidence roles and no explicit exposure or resolution.
"""
    raw = call_llm_json(SYSTEM_PROMPT, user_message)

    scores = {key: score(raw, key) for key in APPROVAL_THRESHOLDS}
    approved = all(
        scores[key] >= minimum
        for key, minimum in APPROVAL_THRESHOLDS.items()
    )
    evidence = raw.get("evidence", []) if is_assumption else []
    status = raw.get("assumption_status") if is_assumption else None
    inference = raw.get("inference", "") if is_assumption else ""
    issues = raw.get("issues", [])
    if is_assumption:
        hard_issues = validate_assumption_evidence(filming.transcript, evidence)
        if status not in {"implicit", "exposed", "resolved", "missing"}:
            hard_issues.append("Missing or invalid assumption status.")
        if not isinstance(inference, str) or not inference.strip():
            hard_issues.append("Missing inference explanation.")
        issues = list(issues) + hard_issues
        approved = approved and status == "implicit" and not hard_issues

    return QAFeedback(
        **scores,
        evidence_turns=([item["turn"] for item in evidence if isinstance(item, dict)
                         and type(item.get("turn")) is int] if is_assumption else [
            t for t in raw.get("evidence_turns", []) if isinstance(t, int)]),
        evidence=evidence if isinstance(evidence, list) else [],
        inference=inference if isinstance(inference, str) else "",
        assumption_status=status if status in {"implicit", "exposed", "resolved", "missing"} else None,
        realism_signals=raw.get("realism_signals", []),
        issues=issues,
        approved=approved,
    )
