"""
Offline tests for multi_meeting.py: the five multi-input tasks run end to
end against the fake LLM from test_task_pipeline.py, extended with the
series planner, the between-meeting summary and the series QA.

    python -m pytest tests/test_multi_meeting.py -q
"""
import json
import random
import re

import pytest

import tests.test_task_pipeline as base
from tests.test_task_pipeline import ROOT, FakeLLM, agent1, agent2, agent3, agent4, style_extractor
import multi_meeting
from prompts import series_prompt


class SeriesFake(FakeLLM):
    def __init__(self, series_scores=5, bad_first_plan=False, **kw):
        super().__init__(term_task=False, **kw)
        self.series_scores = series_scores
        self.bad_first_plan = bad_first_plan
        self.contexts = []

    def json(self, system, user, max_tokens=4000, model=None):
        body = user if isinstance(user, str) else "\n".join(b["text"] for b in user)

        if system.startswith(series_prompt.SERIES_PLAN_PROMPT):
            self.calls["series_plan"] += 1
            req = json.loads(body)
            kind = next(k for k, v in series_prompt.STRUCTURES.items() if v == req["STRUCTURE"])
            n = multi_meeting.MEETING_COUNTS[kind]
            meetings = [{"label": f"Meeting {i + 1} — Feb {i + 1}, 2027",
                         "meeting_context": f"context {i + 1}",
                         "instruction": f"instruction {i + 1}",
                         "ground_truth": {"fact": f"value {i + 1}"},
                         "transcript": "scripted"} for i in range(n)]
            if kind == "paired_meetings":
                meetings[0]["disciplines"] = ["hydrology", "economics", "anthropology"]
                meetings[1]["disciplines"] = ["cognitive neuroscience", "computational neuroscience"]
                if self.bad_first_plan and self.calls["series_plan"] == 1:
                    meetings[1]["disciplines"] = ["neuroscience"]
            plan = {"research_problem": "flood risk", "project_context": "Durham team.",
                    "meetings": meetings, "ground_truth": {"answer": "x"}}
            if kind == "plan_and_meeting":
                plan["plan_document"] = "RQ1 ... Task 1 ..."
            return plan

        if system.startswith(series_prompt.SUMMARY_PROMPT):
            self.calls["summary"] += 1
            return {"summary": "They agreed the plan."}

        if system.startswith(series_prompt.SERIES_QA_PROMPT):
            self.calls["series_qa"] += 1
            self.series_qa_input = body
            return {"series_recoverable": self.series_scores, "series_compliance": "5", "issues": []}

        if system.startswith(base.agent2_prompt.SYSTEM_PROMPT):
            # Agent 1 doesn't run here, so take the disciplines from the casting request.
            self.disciplines = re.findall(r"'discipline': '([^']+)'", body)

        if "Prepare shared meeting background" in system:
            self.contexts.append(json.loads(body)["meeting_context"])

        return super().json(system, user, max_tokens, model)


def run_series(task_id, **kw):
    random.seed(0)
    fake = SeriesFake(**kw)
    patched = [(m, "call_llm_json") for m in (agent1, agent2, agent3, agent4, multi_meeting)]
    saved = {(m, a): getattr(m, a) for m, a in patched}
    saved[(style_extractor, "call_llm")] = style_extractor.call_llm
    for m, a in patched:
        setattr(m, a, fake.json)
    style_extractor.call_llm = fake.text
    try:
        out = multi_meeting.run_series({**base.main.CONFIG, "task_id": task_id, "disciplines": None})
    finally:
        for (m, a), f in saved.items():
            setattr(m, a, f)
    for path in (ROOT / "data" / "outputs").glob(f"{task_id}/*/*_{out['series_id']}.json"):
        saved_json = json.loads(path.read_text())
        path.unlink()
    assert saved_json["series_id"] == out["series_id"]
    return out, fake


@pytest.mark.parametrize("task_id,n", [
    ("problem_02", 1), ("integration_02", 2), ("self_review_01", 3),
    ("adaptation_01", 2), ("other_02", 1)])
def test_each_multi_input_task_runs(task_id, n):
    out, fake = run_series(task_id)
    assert len(out["meetings"]) == n
    assert all(m["transcript"] for m in out["meetings"])
    assert all("transcript" not in m["ground_truth"] for m in out["meetings"])  # scripted dialogue stripped
    assert out["approved"] and fake.calls["series_qa"] == 1


def test_sequence_carries_earlier_meetings_forward():
    out, fake = run_series("self_review_01")
    assert fake.calls["summary"] == 2
    assert "earlier meetings" not in fake.contexts[0]
    assert "Meeting 1 — Feb 1, 2027: They agreed the plan." in fake.contexts[-1]
    # same people across a project's meetings
    names = {tuple(s["name"] for s in m["speakers"]) for m in out["meetings"]}
    assert len(names) == 1


def test_paired_meetings_have_separate_groups():
    out, fake = run_series("integration_02")
    a, b = out["meetings"]
    assert len(a["speakers"]) == 3 and len(b["speakers"]) == 2
    assert fake.calls["summary"] == 0


def test_plan_document_reaches_meeting_and_output():
    out, fake = run_series("problem_02")
    assert out["plan_document"] and "RQ1" in fake.contexts[0]


def test_seniority_versions_swap_labels_only():
    out, _ = run_series("other_02")
    a, b = out["versions"]
    assert a["labels"].keys() == b["labels"].keys()
    for name in a["labels"]:
        assert a["labels"][name] != b["labels"][name]
    roles = sorted(s["functional_role"] for s in out["meetings"][0]["speakers"])
    assert roles == sorted([multi_meeting.SENIOR.value, multi_meeting.JUNIOR.value])


def test_low_series_score_fails_the_sample():
    out, _ = run_series("adaptation_01", series_scores=3)
    assert not out["approved"]


def test_cli_routes_multi_input_tasks(monkeypatch):
    called = {}
    def fake_run(cfg):
        called["cfg"] = cfg
        return {"approved": True}
    monkeypatch.setattr(multi_meeting, "run_series", fake_run)
    monkeypatch.setattr("sys.argv", ["main.py", "--task", "adaptation_01"])
    base.main.cli()
    assert called["cfg"]["task_id"] == "adaptation_01"


def test_bad_paired_plan_is_replanned_before_filming():
    out, fake = run_series("integration_02", bad_first_plan=True)
    assert fake.calls["series_plan"] == 2
    assert len(out["meetings"][1]["speakers"]) == 2


def test_seniority_part_tells_qa_labels_come_later():
    out, _ = run_series("other_02")
    assert "labels are attached to the speakers afterwards" in out["meetings"][0]["seed_spec"]["benchmark_task"]


def test_set_approval_uses_craft_scores_per_meeting():
    from multi_meeting import meeting_craft_ok
    good_craft_weak_seed = {"seed_recoverable": 3, "task_compliance": 3, "naturalness": 3,
                            "speaker_distinctness": 4, "domain_alignment": 4, "dialogue_flow": 4}
    assert meeting_craft_ok(good_craft_weak_seed)
    assert not meeting_craft_ok(dict(good_craft_weak_seed, naturalness=2))
