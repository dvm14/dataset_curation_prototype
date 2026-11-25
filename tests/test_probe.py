"""
Offline tests for the blind probe (agents/agent6_probe.py) and its contrast
with the critic. Every LLM call is a fake.

    python -m pytest tests/test_probe.py -q
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENAI_API_KEY", "offline-test")

import agents.agent5_critic as critic  # noqa: E402
import agents.agent6_probe as probe  # noqa: E402
from benchmark_tasks import TASKS  # noqa: E402
from probe_questions import PROBE_QUESTIONS, CONTROL_TASKS  # noqa: E402
from schema import ProbeAttempt, ProbeReport  # noqa: E402

MEETING = [
    {"speaker": "Ana", "text": "For exposure I mean the share of residents within 800 meters."},
    {"speaker": "Ben", "text": "In my lab exposure is the light the sensor records."},
    {"speaker": "Ana", "text": "Good, so we measure exposure in March."},
]
CAST = [{"name": "Ana", "discipline": "epidemiology"},
        {"name": "Ben", "discipline": "computer science"}]
TRUTH = {"term": "exposure", "secret_marker": "SENSE-SPLIT-XYZ"}
TASK_TEXT = "Give a meeting where researchers use the same term ... UNIQUE-TASK-TEXT"
POINTS = ["Epidemiology uses exposure as proximity", "CS uses exposure as sensor light"]


class Fakes:
    """Solver text replies and grader JSON replies, in order; records what each saw."""

    def __init__(self, answers, grades, critic_reply=None):
        self.answers, self.grades = list(answers), list(grades)
        self.critic_reply = critic_reply
        self.solver_seen, self.grader_seen, self.models = [], [], []

    def text(self, system, user, max_tokens=4000, model=None, json_mode=False):
        self.solver_seen.append(user)
        self.models.append(model)
        reply = self.answers.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def json(self, system, user, max_tokens=4000, model=None):
        if system.startswith(critic.SYSTEM_PROMPT):
            return self.critic_reply
        self.grader_seen.append(user)
        return self.grades.pop(0)

    def install(self, monkeypatch):
        monkeypatch.setattr(probe, "call_llm", self.text)
        monkeypatch.setattr(probe, "call_llm_json", self.json)
        monkeypatch.setattr(critic, "call_llm_json", self.json)


def grade(*statuses, extra=(), false_alarm=None, score=4):
    return {"points": [{"point": p, "status": s} for p, s in zip(POINTS, statuses)],
            "extra_claims": list(extra), "false_alarm": false_alarm, "score": score}


# ── questions ────────────────────────────────

def test_every_single_meeting_task_has_a_question():
    single = {k for k, t in TASKS.items() if t["input_type"] == "single_meeting"}
    assert single == set(PROBE_QUESTIONS)


def test_questions_do_not_give_answers_away():
    for task_id, q in PROBE_QUESTIONS.items():
        text = q.lower()
        assert "evaluated ai" not in text and "control" not in text, task_id
        assert "unnecessary" not in text and "unnoticed" not in text, task_id
    # controls ask the same kind of question as a real task, with "none" allowed
    for task_id in CONTROL_TASKS:
        assert "none" in PROBE_QUESTIONS[task_id].lower() or "if so" in PROBE_QUESTIONS[task_id].lower()


# ── the solver is blind ──────────────────────

def test_solver_sees_only_transcript_and_question(monkeypatch):
    fakes = Fakes(["Answer citing turn 2."], [grade("found", "found")])
    fakes.install(monkeypatch)
    probe.run("jargon_01", MEETING, CAST, TRUTH, points=POINTS)
    seen = fakes.solver_seen[0]
    assert "SENSE-SPLIT-XYZ" not in seen and "secret_marker" not in seen
    assert "epidemiology" not in seen and "computer science" not in seen   # no cast disciplines
    assert PROBE_QUESTIONS["jargon_01"] in seen and MEETING[1]["text"] in seen
    assert "SENSE-SPLIT-XYZ" in fakes.grader_seen[0]                      # the grader does see it
    assert fakes.models[0] == probe.PROBE_MODEL


def test_grader_points_line_up_with_critic_wording(monkeypatch):
    shuffled = {"points": [{"point": POINTS[1], "status": "missed"},
                           {"point": POINTS[0], "status": "found"}], "score": 3}
    Fakes(["a"], [shuffled]).install(monkeypatch)
    report = probe.run("jargon_01", MEETING, CAST, TRUTH, points=POINTS)
    assert report.found_rate == {POINTS[0]: 1.0, POINTS[1]: 0.0}


def test_found_rate_and_false_alarms_across_runs(monkeypatch):
    Fakes(["a", "b"], [grade("found", "partial", false_alarm=True, score=4),
                       grade("missed", "found", false_alarm=False, score=2)]).install(monkeypatch)
    report = probe.run("jargon_03", MEETING, CAST, TRUTH, points=POINTS, runs=2)
    assert report.found_rate == {POINTS[0]: 0.5, POINTS[1]: 0.75}
    assert report.mean_score == 3.0 and report.false_alarm_rate == 0.5


def test_solver_error_is_recorded_not_raised(monkeypatch):
    Fakes([RuntimeError("429"), "ok"], [grade("found", "found")]).install(monkeypatch)
    report = probe.run("jargon_01", MEETING, CAST, TRUTH, points=POINTS, runs=2)
    assert report.attempts[0].error.startswith("solver") and not report.attempts[1].error
    assert report.found_rate == {POINTS[0]: 1.0, POINTS[1]: 1.0}


# ── contrast ─────────────────────────────────

def report_with(found, false_alarm_rate=None, extra=()):
    return ProbeReport(question="q", solver_model="m", grader_model="m",
                       attempts=[ProbeAttempt(run=1, answer="a", extra_claims=list(extra))],
                       found_rate=found, false_alarm_rate=false_alarm_rate)


def test_contrast_kinds():
    points = [{"point": "A", "status": "supported"}, {"point": "B", "status": "missing"},
              {"point": "C", "status": "supported"}]
    rival = {"claim": "the March date is the post-test", "supported_by_transcript": True,
             "conflicts_with_reference": True}
    out = probe.contrast("jargon_02", points,
                         report_with({"A": 0.0, "B": 1.0, "C": 1.0}, extra=[rival]))
    assert [c["kind"] for c in out] == ["too_faint", "critic_disputed", "ambiguous"]


def test_controls_contrast_only_false_alarms():
    points = [{"point": "A", "status": "supported"}]
    rival = {"claim": "x is undefined", "supported_by_transcript": True, "conflicts_with_reference": True}
    out = probe.contrast("jargon_03", points, report_with({"A": 0.0}, false_alarm_rate=1.0, extra=[rival]))
    assert [c["kind"] for c in out] == ["false_alarm_bait"]
    assert probe.contrast("jargon_03", points, report_with({"A": 0.0}, false_alarm_rate=0.0)) == []


def test_not_asked_points_are_left_out_of_found_rate(monkeypatch):
    Fakes(["a"], [grade("found", "not_asked")]).install(monkeypatch)
    report = probe.run("jargon_01", MEETING, CAST, TRUTH, points=POINTS)
    assert report.found_rate == {POINTS[0]: 1.0}


def test_contrast_empty_when_readings_agree():
    points = [{"point": "A", "status": "supported"}, {"point": "B", "status": "missing"}]
    assert probe.contrast("jargon_01", points, report_with({"A": 1.0, "B": 0.0})) == []


# ── critic + probe together ──────────────────

def test_critic_runs_probe_and_flags_a_faint_cue(monkeypatch):
    critic_reply = {"points": [
        {"point": POINTS[0], "status": "supported",
         "evidence": [{"turn": 1, "quote": "the share of residents within 800 meters"}]},
        {"point": POINTS[1], "status": "supported",
         "evidence": [{"turn": 2, "quote": "exposure is the light the sensor records"}]}],
        "seed_recoverable": 5}
    fakes = Fakes(["The team uses exposure one way."], [grade("found", "missed")], critic_reply)
    fakes.install(monkeypatch)
    report = critic.run("jargon_01", TASK_TEXT, TRUTH, MEETING, CAST, probe_runs=1)
    assert report.verdict == "flag"
    assert [c["kind"] for c in report.contrast] == ["too_faint"]
    assert "UNIQUE-TASK-TEXT" not in fakes.solver_seen[0]
    assert report.probe["found_rate"][POINTS[1]] == 0.0
    # the grader graded the critic's own points
    assert all(p in fakes.grader_seen[0] for p in POINTS)
    json.dumps(report.model_dump())                     # serializable for critic.py


def test_probe_off_by_default(monkeypatch):
    fakes = Fakes([], [], {"points": [], "seed_recoverable": 5})
    fakes.install(monkeypatch)
    report = critic.run("jargon_01", TASK_TEXT, TRUTH, MEETING, CAST)
    assert report.probe is None and fakes.solver_seen == []
