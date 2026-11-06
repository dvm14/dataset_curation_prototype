"""
Offline end-to-end test of the task-based pipeline: every LLM call is
replaced by a fake that recognises which prompt it was sent, so the whole
flow (Agent 1 -> QA -> save) runs without an API key or cost.

    python -m pytest tests/test_task_pipeline.py -q
    python tests/test_task_pipeline.py        # also prints call counts

It checks wiring and the code-side rules, not output quality.
"""
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENAI_API_KEY", "offline-test")  # client is built at import

import main  # noqa: E402
import agents.agent1_content_curation as agent1  # noqa: E402
import agents.agent2_casting as agent2  # noqa: E402
import agents.agent3_filming as agent3  # noqa: E402
import agents.agent4_qa as agent4  # noqa: E402
import utils.style_extractor as style_extractor  # noqa: E402
from prompts import agent1_prompt, agent2_prompt, agent3_prompt, agent4_prompt  # noqa: E402

TERM = "resilience"


class FakeLLM:
    """Answers each prompt type with minimal valid JSON and counts calls."""

    def __init__(self, term_task, malformed_leak_every=0, extra_cast=0):
        self.term_task = term_task
        self.extra_cast = extra_cast
        self.malformed_leak_every = malformed_leak_every
        self.calls = Counter()
        self.leak_chars = []
        self.disciplines = []
        self.seed_speakers = []

    def text(self, system, user):
        self.calls["style"] += 1
        return "Short turns, lots of hedging."

    def json(self, system, user, max_tokens=4000, model=None):
        body = user if isinstance(user, str) else "\n".join(b["text"] for b in user)

        if system.startswith(agent1_prompt.SYSTEM_PROMPT):
            self.calls["agent1"] += 1
            self.disciplines = re.search(r"Disciplines in the meeting: (.*)", body).group(1).split(", ")
            truth = ({"term": TERM, "meanings": [
                          {"discipline": self.disciplines[0], "meaning": "x"},
                          {"discipline": self.disciplines[1], "meaning": "y"}],
                      "transcript": [{"speaker": "Speaker C", "utterance": "scripted"}]}
                     if self.term_task else {"conflict": "approval after interviews"})
            return {"research_problem": "urban heat", "disciplines": self.disciplines,
                    "discipline_distance": "distant", "meeting_context": "Week 3 sync.",
                    "seed": {"explicit_instruction": f"{self.disciplines[0]} and "
                             f"{self.disciplines[1]} both use the term.", "ground_truth": truth}}

        if system.startswith(agent2_prompt.SYSTEM_PROMPT):
            self.calls["agent2"] += 1
            cast = [{"name": f"Speaker {chr(65 + i)}", "discipline": d,
                     "background": "x", "distinct_perspective": "y"}
                    for i, d in enumerate(self.disciplines)]
            # extra people cast for roles named in the seed ("project lead")
            cast += [{"name": f"Extra {i}", "discipline": "project lead",
                      "background": "x", "distinct_perspective": "y"}
                     for i in range(self.extra_cast)]
            return {"speakers": cast}

        if "Prepare shared meeting background" in system:
            self.calls["planner"] += 1
            names = [s["name"] for s in json.loads(body)["speakers"]]
            facts = [] if self.term_task else [
                {"speaker": names[0], "fact": "Approval is expected March 1, 2027.", "cue": "March 1, 2027"},
                {"speaker": names[1], "fact": "Interviews start February 1, 2027.", "cue": "February 1, 2027"},
                {"speaker": "Nobody", "fact": "dropped: unknown speaker", "cue": "x"}]
            return {"shared_context": "Durham, grant due Dec 1.",
                    "briefs": {n: f"brief for {n}" for n in names},
                    "required_facts": facts}

        if system.startswith(agent3_prompt.BID_PROMPT):
            self.calls["bid"] += 1
            turns = body.count("\nSpeaker ") + body.startswith("Speaker ")
            return {"urgency": random.randint(5, 9) if turns < 18 else 1,
                    "intent": random.choice(["respond", "respond", "brief", "interject"]),
                    "react": random.random() < 0.5}

        if system.startswith(agent3_prompt.SYSTEM_PROMPT):
            self.calls["speak"] += 1
            name = re.search(r"Speak only as (.+?)\.\n", body).group(1)
            # Seed speakers (first two) only say the term once nudged.
            says_term = self.term_task and name != "Speaker C" and "you haven't raised yet" in body
            filler = "so I think the corridor data from last spring would let us test that properly"
            # Required facts are only said once nudged (with the "1st" form).
            fact = re.search(r"haven't mentioned yet:\n- .*?(\w+) 1, 2027", body)
            return {"text": f"{name} here, {filler}"
                            + (f" and the {TERM} of the system matters" if says_term else "")
                            + (f", and {fact.group(1)} 1st, 2027 is the date" if fact else "") + "."}

        if system.startswith(agent3_prompt.LEAK_CHECK_PROMPT):
            self.calls["leak"] += 1
            self.leak_chars.append(len(body))
            if self.malformed_leak_every and self.calls["leak"] % self.malformed_leak_every == 0:
                return {"leaked": "maybe"}  # malformed on purpose
            return {"leaked": False, "reason": "fine"}

        if system.startswith(agent3_prompt.SPEECH_PASS_PROMPT):
            self.calls["speech"] += 1
            return {"text": json.loads(body)["turn"]}

        if "inspect a generated meeting transcript" in system:
            self.calls["locator"] += 1
            return {"seed_location": "turns 9-12"}

        if system.startswith(agent4_prompt.SYSTEM_PROMPT):
            self.calls["judge"] += 1
            self.judge_message = body
            return {k: 5 for k in agent4.APPROVAL_THRESHOLDS} | {"issues": [], "evidence_turns": [9]}

        if system.startswith(agent4_prompt.SEED_USE_PROMPT):
            self.calls["seed_use"] += 1
            turns = re.findall(r"^\[(\d+)\] [^(]*\(([^)]*)\)", body, re.M)
            return {"turns": [{"turn": int(n), "sense": d, "comments_on_term": False,
                               "renames": False} for n, d in turns]}

        if system.startswith(agent4_prompt.SPOKEN_CHECK_PROMPT):
            self.calls["spoken"] += 1
            return {"labels": []}

        raise AssertionError(f"Unrecognised prompt: {system[:80]!r}")


def run_task(task_id, **fake_kwargs):
    random.seed(0)
    task = main.get_task(task_id)
    fake = FakeLLM(term_task=agent1.is_term_task(task["benchmark_category"], task["benchmark_task"]),
                   **fake_kwargs)
    patched = [(m, "call_llm_json") for m in (agent1, agent2, agent3, agent4)]
    saved = {(m, a): getattr(m, a) for m, a in patched}
    saved[(style_extractor, "call_llm")] = style_extractor.call_llm
    for m, a in patched:
        setattr(m, a, fake.json)
    style_extractor.call_llm = fake.text
    try:
        result = main.run_pipeline({**main.CONFIG, "task_id": task_id, "disciplines": None})
    finally:
        for (m, a), f in saved.items():
            setattr(m, a, f)
    # don't leave fake transcripts in data/outputs
    for path in (ROOT / "data" / "outputs").glob(f"*/*_{result.transcript_id}.json"):
        path.unlink()
    return result, fake


def test_jargon_restores_term_checks():
    result, fake = run_task("jargon_01")
    terms = [TERM]
    said_by = {t.speaker for t in result.transcript if TERM in t.text}
    assert {"Speaker A", "Speaker B"} <= said_by, "nudges should get both seed speakers to say the term"
    # leak checks only run on turns that contain the term
    assert fake.calls["leak"] <= sum(TERM in t.text for t in result.transcript) * 2 + 2
    assert "SEED TERM USES" in fake.judge_message
    assert fake.calls["seed_use"] == 1 and result.qa_feedback.seed_uses
    assert result.ground_truth["term"] == TERM and terms


def test_control_and_other_tasks_skip_term_checks():
    for task_id in ("jargon_03", "coordination_03"):
        result, fake = run_task(task_id)
        assert fake.calls["seed_use"] == 0
        assert "SEED TERM USES" not in fake.judge_message
        assert not any(t.text.endswith("—") for t in result.transcript), "no cut-offs outside jargon"
        # every turn and rewrite is checked, but with a bounded window
        assert fake.calls["leak"] >= len(result.transcript) - 2
        assert max(fake.leak_chars) < 6000


def test_malformed_leak_reply_does_not_crash():
    result, _ = run_task("coordination_03", malformed_leak_every=3)
    assert result.transcript


def test_task_defaults_apply():
    result, _ = run_task("assumptions_02")
    assert result.disciplines == ["clinical medicine", "sociology"]
    assert len(result.speakers) == 2


if __name__ == "__main__":
    for task_id in ("jargon_01", "jargon_03", "coordination_03"):
        result, fake = run_task(task_id)
        print(f"{task_id:16} turns={len(result.transcript):2}  calls={dict(fake.calls)}  "
              f"max_leak_input={max(fake.leak_chars or [0]):,} chars")
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)


def test_seed_speakers_ignore_scripted_example():
    """Regression, run 6c88f13b: a third speaker named only in a scripted
    example inside ground_truth was treated as a seed speaker, and QA capped
    task_compliance because 'the economics sense never appears'."""
    from types import SimpleNamespace as NS
    from utils.seed_terms import seed_speakers

    truth = {"term": "attention", "meanings": [
        {"discipline": "neuroscience", "speaker": "Dr. Elena Park", "meaning": "cognitive focus"},
        {"discipline": "political science", "speaker": "Dr. Malik Torres", "meaning": "issue salience"}],
        "transcript": [{"speaker": "Dr. Priya Shah, economist", "utterance": "I will keep those separate."}]}
    speakers = [NS(name="Dr. Elena Park", discipline="neuroscience"),
                NS(name="Dr. Malik Torres", discipline="political science"),
                NS(name="Dr. Priya Shah", discipline="economics")]
    content = NS(seed=NS(ground_truth=truth, explicit_instruction="Read the meeting transcript.",
                         benchmark_category="Jargon", benchmark_task="Give a meeting..."))
    assert [s.name for s in seed_speakers(content, speakers)] == ["Dr. Elena Park", "Dr. Malik Torres"]

    cleaned = agent1.strip_dialogue(truth)
    assert "transcript" not in cleaned and cleaned["term"] == "attention"


def test_required_facts_get_said_before_meeting_ends():
    result, fake = run_task("coordination_03")
    text = " ".join(t.text for t in result.transcript)
    assert "March 1st, 2027" in text and "February 1st, 2027" in text


def test_extra_cast_is_trimmed_not_fatal():
    result, _ = run_task("coordination_03", extra_cast=3)
    assert [s.name for s in result.speakers] == ["Speaker A", "Speaker B", "Speaker C"]


def test_cue_matching():
    from utils.required_facts import cue_said, parse_facts
    assert cue_said("so Feb- February 1st, 2027 works", "February 1, 2027")
    assert not cue_said("February 10, 2027", "February 1, 2027")
    assert not cue_said("in February", "February 1, 2027")
    from types import SimpleNamespace as NS
    facts = parse_facts([{"speaker": "A", "fact": "f", "cue": "one two three four five six seven"},
                         {"speaker": "A", "fact": "f", "cue": "Priya"}, "junk"], [NS(name="A")])
    assert [f["cue"] for f in facts] == ["Priya"]
