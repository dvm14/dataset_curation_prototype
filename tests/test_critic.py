"""
Offline tests for the critic: code checks, quote verification, verdicts and
the batch CLI. The critic's LLM call is replaced by a fake.

    python -m pytest tests/test_critic.py -q
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENAI_API_KEY", "offline-test")

import agents.agent5_critic as critic  # noqa: E402
import critic as critic_cli  # noqa: E402
from utils import critic_checks as checks  # noqa: E402


def turns(*pairs):
    return [{"speaker": s, "text": t} for s, t in pairs]


MEETING = turns(
    ("Ana", "Hi everyone, thanks for making the time. Before we get into it, are we all here?"),
    ("Ben", "Yes. So for exposure I mean the share of residents within 800 meters of the park."),
    ("Cai", "In my lab exposure is the light the sensor records, actually, actually the integrated dose."),
    ("Ana", "Ms. Brooks said the liaison office wants surveys only, no passive location data."),
    ("Ben", "Then we go with GPS tracking for the whole cohort starting in March, that's the plan."),
    ("Cai", "I'll draft the protocol by Friday and send it to everyone for comments."),
    ("Ana", "Okay."),
    ("Ben", "Sounds good."),
    ("Ben", "Talk soon."),
)
CAST = [{"name": "Ana Ruiz", "discipline": "ecology"},
        {"name": "Ben Ode", "discipline": "epidemiology"},
        {"name": "Cai Lin", "discipline": "computer science"}]


# ── code checks ──────────────────────────────

def test_doubled_words_skips_grammatical_doubles():
    found = checks.doubled_words(turns(("A", "I think that that works, very, very well. Not, not yet.")))
    assert [h["text"].lower() for h in found["hits"]] == ["very, very", "not, not"]


def test_stock_opening_and_uncast_names():
    result = checks.run_checks(MEETING, [c["name"] for c in CAST])
    assert result["stock_opening"]["flag"]
    assert result["uncast_names"]["names"] == ["Ms. Brooks"]


def test_strict_rotation_and_even_counts():
    rotation = turns(*[(s, "a substantive point about the study design here") for s in "ABCABCABC"])
    assert checks.turn_balance(rotation)["strict_rotation"]
    uneven = turns(*[(s, "x") for s in "AABACAAB"])
    balance = checks.turn_balance(uneven)
    assert not balance["strict_rotation"] and not balance["even_counts"]


def test_tidy_ending_ignores_goodbyes():
    long = "and that covers the analysis plan we discussed earlier today in detail"
    ending = turns(
        ("A", f"I'll send the consent form by Friday {long}"),
        ("B", f"I will update the sampling frame next week {long}"),
        ("C", f"We'll circulate the action items with owners {long}"),
        ("A", "Thanks, everyone. Bye."),
    )
    result = checks.tidy_ending(ending)
    assert result["flag"] and result["commitment_turns"] == [1, 2, 3] and result["goodbye"]


def test_repeated_openers_ignore_common_starts():
    body = "ever come through with the model numbers from the vendor last week"
    meeting = turns(*[(s, f"Did the file {body}") for s in "ABA"],
                    *[(s, f"I think we should {body}") for s in "BAB"])
    assert list(checks.repeated_openers(meeting)["repeated"]) == ["did the"]


# ── quote verification ───────────────────────

def test_find_quote_tolerates_punctuation_and_fixes_turn():
    assert critic.find_quote("Then we go with GPS tracking", 5, MEETING) == 5
    assert critic.find_quote("surveys only no passive location data", 1, MEETING) == 4
    assert critic.find_quote("we will use satellite imagery", 2, MEETING) is None


# ── verdicts with a fake critic model ────────

class FakeCritic:
    def __init__(self, reply):
        self.reply = reply
        self.messages = []

    def __call__(self, system, user, max_tokens=4000, model=None):
        self.messages.append(user)
        return self.reply


def run_with(monkeypatch, task_id, reply):
    fake = FakeCritic(reply)
    monkeypatch.setattr(critic, "call_llm_json", fake)
    report = critic.run(task_id, "task text", {"answer": "x"}, MEETING, CAST)
    return report, fake


def point(status="supported", quote="Then we go with GPS tracking", turn=5, **extra):
    return {"point": "the team picks GPS", "status": status,
            "evidence": [{"turn": turn, "quote": quote}], **extra}


def test_clean_item_passes(monkeypatch):
    report, fake = run_with(monkeypatch, "problem_01", {
        "points": [point()], "seed_recoverable": 5, "seed_implicit": 5})
    # Code flags here: stock opening and an uncast name, under the review limit of 3.
    assert report.verdict == "pass", report.reasons
    assert report.seed_implicit is None            # nothing must stay unnoticed
    assert "MUST STAY UNNOTICED" not in fake.messages[0]


def test_fabricated_evidence_is_dropped_and_flagged(monkeypatch):
    report, _ = run_with(monkeypatch, "problem_01", {
        "points": [point(quote="we agreed on satellite imagery")], "seed_recoverable": 5})
    assert report.unverified_quotes == 1
    assert report.points[0]["status"] == "unverified"
    assert report.verdict == "flag"


def test_absence_claim_needs_no_quote(monkeypatch):
    absent = {"point": "nobody notices the mismatch", "status": "supported",
              "evidence": [], "absence": True}
    report, _ = run_with(monkeypatch, "problem_01", {"points": [absent], "seed_recoverable": 5})
    assert report.points[0]["status"] == "supported" and report.verdict == "pass"


def test_surfaced_hidden_issue_blocks(monkeypatch):
    report, fake = run_with(monkeypatch, "assumptions_03", {
        "points": [point()],
        "surfaced": [{"turn": 5, "quote": "surveys only, no passive location", "how": "named"}],
        "seed_recoverable": 5, "seed_implicit": 2})
    assert "MUST STAY UNNOTICED" in fake.messages[0]
    assert report.surfaced[0]["turn"] == 4         # corrected to where the quote is
    assert report.verdict == "block"
    assert any("surfaced" in r for r in report.reasons)
    assert any("seed_implicit 2" in r for r in report.reasons)


def test_unresolved_task_blocks_only_on_resolution(monkeypatch):
    discussed = {"turn": 2, "quote": "for exposure I mean the share of residents", "how": "named"}
    settled = {"turn": 5, "quote": "that's the plan", "how": "resolved"}
    report, fake = run_with(monkeypatch, "assumptions_04", {
        "points": [point()], "surfaced": [discussed], "seed_recoverable": 5, "seed_implicit": 5})
    assert "MUST STAY UNRESOLVED" in fake.messages[0]
    assert report.verdict == "pass", report.reasons

    report, _ = run_with(monkeypatch, "assumptions_04", {
        "points": [point()], "surfaced": [discussed, settled], "seed_recoverable": 5, "seed_implicit": 4})
    assert report.verdict == "block"
    assert report.reasons == ["The hidden issue is resolved in the meeting (turns 5)."]


def test_surfaced_on_unhidden_task_is_ignored(monkeypatch):
    report, _ = run_with(monkeypatch, "coordination_01", {
        "points": [point()],
        "surfaced": [{"turn": 4, "quote": "surveys only", "how": "named"}],
        "seed_recoverable": 5, "seed_implicit": 1})
    assert report.verdict == "pass" and report.seed_implicit is None


def test_control_violation_blocks(monkeypatch):
    report, fake = run_with(monkeypatch, "jargon_03", {
        "points": [point()],
        "control_violations": [{"turn": 3, "quote": "exposure is the light the sensor records",
                                "problem": "second sense"}],
        "seed_recoverable": 5})
    assert "CONTROL" in fake.messages[0]
    assert report.verdict == "block"


def test_partial_point_and_missing_role_flag(monkeypatch):
    report, _ = run_with(monkeypatch, "problem_01", {
        "points": [point(status="partial", needs_outside_knowledge=True)],
        "missing_participants": ["project manager"], "seed_recoverable": 4})
    assert report.verdict == "flag"
    assert any("project manager" in r for r in report.reasons)


def test_low_seed_recoverable_blocks(monkeypatch):
    report, _ = run_with(monkeypatch, "problem_01", {"points": [point()], "seed_recoverable": 3})
    assert report.verdict == "block"


def test_no_llm_runs_code_checks_only(monkeypatch):
    monkeypatch.setattr(critic, "call_llm_json", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    report = critic.run("problem_01", "t", {}, MEETING, CAST, use_llm=False)
    assert report.checks["uncast_names"]["names"] == ["Ms. Brooks"]
    assert report.seed_recoverable is None


# ── batch CLI ────────────────────────────────

def saved(path, task, status, transcript=MEETING):
    data = {"task_id": task, "benchmark_task": "t", "ground_truth": {"a": 1},
            "transcript": transcript, "speakers": CAST,
            "qa_feedback": {"seed_recoverable": 5, "approved": status == "approved"}}
    target = path / task / status / f"{task}_{status}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data))


def test_cli_skips_multi_meeting_and_summarizes(tmp_path, monkeypatch):
    saved(tmp_path, "problem_01", "approved")
    saved(tmp_path, "jargon_03", "failed")
    multi = tmp_path / "adaptation_01" / "approved" / "set.json"
    multi.parent.mkdir(parents=True)
    multi.write_text(json.dumps({"task_id": "adaptation_01", "meetings": []}))

    items, skipped = critic_cli.load_items(tmp_path)
    assert [d["task_id"] for _, d in items] == ["jargon_03", "problem_01"] and skipped == 1

    monkeypatch.setattr(critic, "call_llm_json", FakeCritic({"points": [point()], "seed_recoverable": 3}))
    rows = [critic_cli.row_for(p, d, critic_cli.critique(d, use_llm=True)) for p, d in items]
    md = critic_cli.summary_markdown(rows, skipped, True, "gpt-5.6-terra")
    assert "| approved | 0 | 0 | 1 |" in md     # QA approved, critic blocks
    assert "| failed | 0 | 0 | 1 |" in md
    assert "1 multi-meeting sets skipped" in md
