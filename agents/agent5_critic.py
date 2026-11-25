"""
agent5_critic.py
Critic — a second pass after QA that looks for reasons an item should not
enter the benchmark.

QA (Agent 4) scores the meeting. In the Week 9-10 read-through, people found
items QA had passed or scored too generously:
- the seed is not in the transcript (assumptions-03, coordination-03), or a
  ground-truth point is never said (problem-01's sample size)
- the issue that must stay unnoticed gets noticed or resolved (assumptions-04,
  disagreement-01), the failure Rita traced to Claude's question rate
- a role the task needs never speaks (assumptions-02's project manager)
- a meaning is only hinted at, so answering needs outside knowledge (jargon-01)

The critic answers those questions with verbatim quotes. Every quote is
checked against the transcript here, and quotes that aren't there are
dropped, so a block always points at real lines. Rubric v2's seed_implicit,
which QA doesn't score, comes from the critic. Code-side checks
(utils/critic_checks.py) are attached as flags and never block on their own.

Uses JUDGE_MODEL (gpt-5.6-terra unless .env says otherwise), so one judge
grades every generator.
"""
from __future__ import annotations

import json
import re

from schema import CriticReport
from utils.llm import call_llm_json, JUDGE_MODEL
from utils.critic_checks import run_checks
from prompts.critic_prompt import SYSTEM_PROMPT

# Tasks where something must stay unnoticed, and what. The critic's
# seed_implicit only applies to these.
MUST_STAY_UNNOTICED = {
    "jargon_02": "the terminology mismatch: nobody notices the two speakers mean different things",
    "assumptions_02": "the difference in standards of evidence: nobody names it",
    "assumptions_03": "the conflict between the early constraint and the later decision: nobody connects them",
    "methods_03": "that the chosen method cannot answer the research question: nobody points it out",
    "coordination_03": "the conflict between the approval date and the interview start date: nobody flags it",
    "opportunity_03": "the missing subtopic: nobody announces that it is missing",
    "disagreement_03": "the two disagreements: nobody states them outright as disagreements",
}

# Tasks where the issue is meant to be talked about but must not be settled.
# Only a turn that resolves it counts against the item. In the first live run
# all five assumptions_04 samples were blocked for discussing the ambiguity,
# which is what the task asks for.
MUST_STAY_UNRESOLVED = {
    "assumptions_04": "which interpretation of the ambiguous assumption is right",
    "problem_03": "the alternative research questions that were not chosen",
    "coordination_02": "the complete next-step plan: nobody lays one out with owners and dates",
    "disagreement_02": "the dissent: the objector does not raise it again and the group does not answer it",
}


def hidden_issue(task_id):
    """(heading, text) to show the critic, or None for tasks with nothing hidden."""
    if task_id in MUST_STAY_UNNOTICED:
        return "MUST STAY UNNOTICED", MUST_STAY_UNNOTICED[task_id]
    if task_id in MUST_STAY_UNRESOLVED:
        return ("MUST STAY UNRESOLVED (discussing it is expected; only a turn that settles "
                "it counts, so list only how=\"resolved\" turns)", MUST_STAY_UNRESOLVED[task_id])
    return None

# Controls: the meeting must not contain the problem.
CONTROLS = {
    "jargon_03": "terminology confusion: a term used in two senses, or a term used without being defined",
    "opportunity_04": "a missing perspective the research problem needs",
    "other_01": "a conflict or misunderstanding that needs an intervention",
}

# Inclusion rule from rubric v2.
INCLUSION_MIN = 4
# Code-side flags that, together, send an item to a human.
FLAGS_FOR_REVIEW = 3
MAX_QUOTE_WORDS = 40


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(r"[–—-]+", " ", text)
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_quote(quote, turn, transcript):
    """
    The turn number where `quote` appears, preferring the cited turn; None
    when it's in no turn. Punctuation, dashes and case are ignored, since
    models re-punctuate when they quote.
    """
    q = _normalize(str(quote or ""))
    if not q or len(q.split()) > MAX_QUOTE_WORDS:
        return None
    texts = [_normalize(t.text if hasattr(t, "text") else t["text"]) for t in transcript]
    if isinstance(turn, int) and 1 <= turn <= len(texts) and q in texts[turn - 1]:
        return turn
    for i, text in enumerate(texts, start=1):
        if q in text:
            return i
    return None


def _verified(items, transcript):
    """Keep the items whose quote is in the transcript, with corrected turn numbers."""
    kept, dropped = [], 0
    for item in items or []:
        if not isinstance(item, dict):
            continue
        turn = find_quote(item.get("quote"), item.get("turn"), transcript)
        if turn is None:
            dropped += 1
            continue
        kept.append({**item, "turn": turn})
    return kept, dropped


def _score(raw, key):
    value = raw.get(key)
    if value is None:
        return None
    try:
        return max(1, min(5, int(value)))
    except (TypeError, ValueError):
        return None


def critic_message(task_id, benchmark_task, ground_truth, transcript, speakers):
    """
    What the critic sees: the task, cast, ground truth and transcript. Not the
    planned moments (beats): those say how the meeting was generated, which the
    critic is told not to use, and they made faint evidence look placed.
    """
    lines = "\n".join(
        f"[{i}] {t.speaker if hasattr(t, 'speaker') else t['speaker']}: "
        f"{t.text if hasattr(t, 'text') else t['text']}"
        for i, t in enumerate(transcript, start=1)
    )
    cast = "\n".join(
        f"- {s['name'] if isinstance(s, dict) else s.name} "
        f"({s['discipline'] if isinstance(s, dict) else s.discipline})"
        for s in speakers
    )
    parts = [
        f"TASK ID: {task_id}",
        f"TASK: {benchmark_task}",
        f"PEOPLE WHO SPEAK IN THE MEETING:\n{cast}",
        f"GROUND TRUTH:\n{json.dumps(ground_truth, indent=2, ensure_ascii=False)}",
    ]
    hidden = hidden_issue(task_id)
    if hidden:
        parts.append(f"{hidden[0]}: {hidden[1]}")
    if task_id in CONTROLS:
        parts.append(f"CONTROL: the meeting must not contain {CONTROLS[task_id]}.")
    parts.append(f"TRANSCRIPT (turn numbers in brackets):\n{lines}")
    return "\n\n".join(parts)


def decide(task_id, points, surfaced, violations, missing, seed_recoverable,
           seed_implicit, check_flags):
    """Verdict and reasons from the verified findings. Pure, so it is tested offline."""
    block, flag = [], []
    for p in points:
        status = p.get("status")
        if status in ("missing", "contradicted"):
            block.append(f"Ground-truth point {status}: {p.get('point', '')}")
        elif status == "unverified":
            # The critic's quote wasn't in the transcript: it may be wrong
            # either way, so a person checks.
            flag.append(f"No verifiable evidence for: {p.get('point', '')}")
        elif status == "partial":
            flag.append(f"Only partly supported: {p.get('point', '')}")
        if p.get("needs_outside_knowledge"):
            flag.append(f"Needs outside knowledge: {p.get('point', '')}")
    if task_id in MUST_STAY_UNRESOLVED:
        surfaced = [s for s in surfaced if s.get("how") == "resolved"]
    if hidden_issue(task_id) and surfaced:
        turns = ", ".join(str(t) for t in sorted({s["turn"] for s in surfaced}))
        what = "resolved" if task_id in MUST_STAY_UNRESOLVED else "surfaced"
        block.append(f"The hidden issue is {what} in the meeting (turns {turns}).")
    if task_id in CONTROLS and violations:
        turns = ", ".join(str(v["turn"]) for v in violations)
        block.append(f"The control contains the problem it must not have (turns {turns}).")
    if seed_recoverable is not None and seed_recoverable < INCLUSION_MIN:
        block.append(f"Critic seed_recoverable {seed_recoverable} is below {INCLUSION_MIN}.")
    if seed_implicit is not None and seed_implicit < INCLUSION_MIN:
        block.append(f"Critic seed_implicit {seed_implicit} is below {INCLUSION_MIN}.")
    for role in missing:
        flag.append(f"Never speaks: {role}")
    if len(check_flags) >= FLAGS_FOR_REVIEW:
        flag.append(f"{len(check_flags)} realism flags from the code checks.")
    verdict = "block" if block else "flag" if flag else "pass"
    return verdict, block + flag


def run(task_id, benchmark_task, ground_truth, transcript, speakers,
        use_llm=True, probe_runs=0) -> CriticReport:
    """
    transcript: list of Turn (or dicts with speaker/text); speakers: list of
    SpeakerProfile (or dicts). With use_llm=False only the code checks run.
    probe_runs > 0 also runs the blind probe (agents/agent6_probe.py) that many
    times and contrasts it with the critic's reading.
    """
    if not transcript:
        raise ValueError("Cannot critique an empty transcript")
    names = [s["name"] if isinstance(s, dict) else s.name for s in speakers]
    checks = run_checks(transcript, names)
    if not use_llm:
        verdict = "flag" if len(checks["flags"]) >= FLAGS_FOR_REVIEW else "pass"
        return CriticReport(verdict=verdict, reasons=checks["flags"], checks=checks)

    raw = call_llm_json(
        SYSTEM_PROMPT,
        critic_message(task_id, benchmark_task, ground_truth, transcript, speakers),
        max_tokens=4000,
        model=JUDGE_MODEL,
    )

    dropped = 0
    points = []
    for p in raw.get("points", []):
        if not isinstance(p, dict):
            continue
        evidence, n = _verified(p.get("evidence"), transcript)
        dropped += n
        status = p.get("status")
        # A claim the critic calls supported but can't quote isn't shown to be
        # supported. Claims that something doesn't happen have nothing to quote.
        if status in ("supported", "partial") and not evidence and p.get("absence") is not True:
            status = "unverified"
        points.append({**p, "status": status, "evidence": evidence})
    surfaced, n = _verified(raw.get("surfaced"), transcript)
    dropped += n
    violations, n = _verified(raw.get("control_violations"), transcript)
    dropped += n
    missing = [str(m) for m in raw.get("missing_participants", []) if m]

    seed_recoverable = _score(raw, "seed_recoverable")
    seed_implicit = _score(raw, "seed_implicit") if hidden_issue(task_id) else None
    verdict, reasons = decide(task_id, points, surfaced, violations, missing,
                              seed_recoverable, seed_implicit, checks["flags"])

    probe, contrast = None, []
    if probe_runs > 0:
        import agents.agent6_probe as agent6
        probe = agent6.run(task_id, transcript, speakers, ground_truth,
                           points=[p["point"] for p in points if p.get("point")],
                           runs=probe_runs)
        contrast = agent6.contrast(task_id, points, probe)
        # The probe only adds flags: a blind reader missing something is a
        # reason for a person to look, not proof the item is broken.
        new = [c["reason"] for c in contrast if c["reason"] not in reasons]
        reasons = reasons + new
        if new and verdict == "pass":
            verdict = "flag"

    return CriticReport(
        probe=probe.model_dump() if probe else None,
        contrast=contrast,
        verdict=verdict,
        reasons=reasons,
        seed_recoverable=seed_recoverable,
        seed_implicit=seed_implicit,
        points=points,
        surfaced=surfaced,
        control_violations=violations,
        missing_participants=missing,
        unverified_quotes=dropped,
        checks=checks,
        judge_model=JUDGE_MODEL,
    )
