"""
agent6_probe.py
Blind probe: answer the benchmark question the way an assistant would, then
grade the answer against the ground truth.

The critic (agent5) reads the transcript knowing the answer, which makes faint
evidence look sufficient. The probe doesn't: the solver gets only the
transcript and the question from probe_questions.py (no ground truth, task
description, cast list, beats or meeting style). A grader that does see the
ground truth then marks each point found / partial / missed / contradicted,
and lists claims beyond it. contrast() puts the two readings side by side:

- the critic finds a point but blind solvers miss it   -> cue too faint
- the critic calls a point missing but solvers find it -> the critic may be wrong
- solvers report a problem in a control                -> false-alarm bait
- solvers give a different answer the transcript backs -> the item is ambiguous

Each contrast is a flag for a person, never a block on its own.

PROBE_MODEL sets the solver (defaults to JUDGE_MODEL). A different or cheaper
model than the judge is closer to the assistants being evaluated. The grader
uses JUDGE_MODEL.
"""
from __future__ import annotations

import json
import os

from schema import ProbeAttempt, ProbeReport
from utils.llm import call_llm, call_llm_json, JUDGE_MODEL
from probe_questions import CONTROL_TASKS, probe_question
from prompts.probe_prompt import SOLVER_PROMPT, GRADER_PROMPT, SPLIT_NOTE

PROBE_MODEL = os.getenv("PROBE_MODEL", JUDGE_MODEL)

FOUND_CREDIT = {"found": 1.0, "partial": 0.5, "missed": 0.0, "contradicted": 0.0}
# Points the question doesn't ask for (a definition when it asks for problems).
# Left out of found_rate: missing them says nothing about the item.
NOT_ASKED = "not_asked"
# A point counts as found by the probe when at least half the credit is there.
FOUND_MIN = 0.5


def _turns_text(transcript):
    return "\n".join(
        f"[{i}] {t.speaker if hasattr(t, 'speaker') else t['speaker']}: "
        f"{t.text if hasattr(t, 'text') else t['text']}"
        for i, t in enumerate(transcript, start=1)
    )


def solver_message(transcript, question):
    """Everything the solver sees. Tested so nothing about the answer leaks in."""
    return f"MEETING TRANSCRIPT (turn numbers in brackets):\n{_turns_text(transcript)}\n\nREQUEST:\n{question}"


def grader_message(task_id, question, ground_truth, points, transcript, answer):
    parts = [
        f"QUESTION THE ASSISTANT GOT:\n{question}",
        f"REFERENCE:\n{json.dumps(ground_truth, indent=2, ensure_ascii=False)}",
    ]
    if points:
        parts.append("REFERENCE POINTS (grade these, same wording and order):\n"
                     + "\n".join(f"- {p}" for p in points))
    else:
        parts.append(SPLIT_NOTE.strip())
    if task_id in CONTROL_TASKS:
        parts.append("CONTROL: this meeting is meant to have no problem of the kind the "
                     "question asks about. Set false_alarm.")
    parts.append(f"TRANSCRIPT:\n{_turns_text(transcript)}")
    parts.append(f"ASSISTANT'S ANSWER:\n{answer}")
    return "\n\n".join(parts)


def _graded_points(raw, points):
    """The grader's points, lined up with the critic's wording when it was given."""
    got = [p for p in raw.get("points", []) if isinstance(p, dict)]
    if not points:
        return [{"point": str(p.get("point", "")), "status": p.get("status", "missed"),
                 "note": p.get("note", "")} for p in got]
    by_text = {str(p.get("point", "")).strip().lower(): p for p in got}
    lined = []
    for i, point in enumerate(points):
        match = by_text.get(point.strip().lower()) or (got[i] if i < len(got) else {})
        status = match.get("status")
        lined.append({"point": point,
                      "status": status if status in FOUND_CREDIT or status == NOT_ASKED else "missed",
                      "note": match.get("note", "")})
    return lined


def _score(value):
    try:
        return max(1, min(5, int(value)))
    except (TypeError, ValueError):
        return None


def attempt(run, task_id, question, transcript, ground_truth, points):
    try:
        answer = call_llm(SOLVER_PROMPT, solver_message(transcript, question),
                          max_tokens=2500, model=PROBE_MODEL)
    except Exception as e:                       # one bad run shouldn't stop the item
        return ProbeAttempt(run=run, answer="", error=f"solver: {e}")
    try:
        raw = call_llm_json(GRADER_PROMPT,
                            grader_message(task_id, question, ground_truth, points,
                                           transcript, answer),
                            max_tokens=2500, model=JUDGE_MODEL)
    except Exception as e:
        return ProbeAttempt(run=run, answer=answer, error=f"grader: {e}")
    false_alarm = raw.get("false_alarm") if task_id in CONTROL_TASKS else None
    return ProbeAttempt(
        run=run,
        answer=answer,
        points=_graded_points(raw, points),
        extra_claims=[c for c in raw.get("extra_claims", []) if isinstance(c, dict)],
        false_alarm=false_alarm if isinstance(false_alarm, bool) else None,
        score=_score(raw.get("score")),
    )


def summarize(question, attempts):
    ok = [a for a in attempts if not a.error]
    found = {}
    for a in ok:
        for p in a.points:
            if p["status"] != NOT_ASKED:
                found.setdefault(p["point"], []).append(FOUND_CREDIT.get(p["status"], 0.0))
    scores = [a.score for a in ok if a.score is not None]
    alarms = [a.false_alarm for a in ok if a.false_alarm is not None]
    return ProbeReport(
        question=question, solver_model=PROBE_MODEL, grader_model=JUDGE_MODEL,
        attempts=attempts,
        found_rate={k: round(sum(v) / len(v), 2) for k, v in found.items()},
        mean_score=round(sum(scores) / len(scores), 2) if scores else None,
        false_alarm_rate=round(sum(alarms) / len(alarms), 2) if alarms else None,
    )


def run(task_id, transcript, speakers, ground_truth, points=None, runs=1) -> ProbeReport:
    """
    points: the critic's ground-truth claims, so both readings grade the same
    list. speakers is accepted for symmetry with the critic but not shown to
    the solver: naming each person's discipline would answer half the
    integration tasks.
    """
    question = probe_question(task_id)
    attempts = [attempt(i, task_id, question, transcript, ground_truth, points or [])
                for i in range(1, runs + 1)]
    return summarize(question, attempts)


def contrast(task_id, critic_points, probe) -> list[dict]:
    """Where the critic (knows the answer) and the blind probe disagree. Pure."""
    if probe is None or not any(not a.error for a in probe.attempts):
        return []
    out = []
    runs = sum(1 for a in probe.attempts if not a.error)
    if task_id in CONTROL_TASKS:
        # A control's only question is whether a blind reader finds a problem
        # that isn't meant to be there; its reference points are background.
        if (probe.false_alarm_rate or 0) >= FOUND_MIN:
            out.append({"kind": "false_alarm_bait", "point": None, "critic": None,
                        "found_rate": probe.false_alarm_rate,
                        "reason": f"Blind solver reported a problem in this control "
                                  f"({probe.false_alarm_rate:.0%} of {runs} runs)."})
        return out
    for p in critic_points:
        point, status = p.get("point", ""), p.get("status")
        rate = probe.found_rate.get(point)
        if rate is None:
            continue
        if status in ("supported", "partial") and rate < FOUND_MIN:
            out.append({"kind": "too_faint", "point": point, "critic": status, "found_rate": rate,
                        "reason": f"Blind solver missed a point the critic found "
                                  f"(found {rate:.0%} of {runs} runs): {point}"})
        elif status in ("missing", "contradicted") and rate >= FOUND_MIN:
            out.append({"kind": "critic_disputed", "point": point, "critic": status,
                        "found_rate": rate,
                        "reason": f"Blind solver found a point the critic called {status}: {point}"})
    rival = sorted({c.get("claim", "") for a in probe.attempts if not a.error
                    for c in a.extra_claims
                    if c.get("supported_by_transcript") is True
                    and c.get("conflicts_with_reference") is True})
    for claim in rival[:3]:
        out.append({"kind": "ambiguous", "point": claim, "critic": None, "found_rate": None,
                    "reason": f"Blind answer differs from the ground truth and the transcript "
                              f"supports it: {claim}"})
    return out
