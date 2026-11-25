"""
Offline tests for the batch variety report (utils/variety.py, variety.py).

    python -m pytest tests/test_variety.py -q
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import variety as cli  # noqa: E402
from utils import variety as v  # noqa: E402

LONG = "we should compare the clinic referral records against the county registry before March"


def single(task, style, research, status="approved", gen="gpt", seed=1, names=("Dr. Maya Chen", "Ben Ode"),
           opening="Thanks everyone for joining today, let's start with the registry",
           question=False):
    turns = [{"speaker": names[0], "text": opening},
             {"speaker": names[1], "text": LONG + ("?" if question else ".")},
             {"speaker": names[0], "text": "Sounds good. Bye."}]
    return v.to_record({
        "task_id": task, "meeting_style": style, "research_id": research,
        "meeting_opening": "late", "generator_model": gen, "sample_seed": seed,
        "disciplines": ["ecology", "economics"], "discipline_distance": "distant",
        "speakers": [{"name": n, "discipline": "x"} for n in names],
        "transcript": turns, "qa_feedback": {"approved": status == "approved"},
    }, status=status)


def test_to_record_handles_multi_meeting_sets():
    rec = v.to_record({"task_id": "adaptation_01", "input_type": "meeting_pair", "approved": True,
                       "meetings": [{"speakers": [{"name": "A"}], "disciplines": ["b", "a"],
                                     "transcript": [{"speaker": "A", "text": "hi"}]}] * 2})
    assert rec["kind"] == "meeting_pair" and rec["status"] == "approved"
    assert len(rec["meetings"]) == 2 and rec["disciplines"] == ("a", "b")


def test_distribution_reports_unused_bank_entries_and_evenness():
    recs = [single("jargon_01", "quick", "r1"), single("jargon_01", "quick", "r2"),
            single("jargon_02", "tangent", "r1")]
    d = v.distribution(recs, "style", universe=["quick", "tangent", "round_robin"])
    assert d["counts"] == {"quick": 2, "tangent": 1} and d["unused"] == ["round_robin"]
    assert 0 < d["evenness"] < 1
    assert v.normalized_entropy({"a": 5, "b": 5}) == 1.0


def test_per_task_repeats_and_distinct():
    recs = [single("jargon_01", "s", "r1") for _ in range(3)] + [single("jargon_01", "s", "r2")]
    assert v.per_task_repeats(recs, "research_id") == [
        {"task_id": "jargon_01", "value": "r1", "count": 3, "of": 4}]
    assert v.per_task_distinct(recs, "research_id") == {"jargon_01": {"distinct": 2, "samples": 4}}


def test_shape_and_question_rate():
    s = v.shape(single("t", "s", "r", question=True)["meetings"][0])
    assert s["turns"] == 3 and s["question_rate"] == round(1 / 3, 3)
    assert s["top_airtime"] == 0.5               # 14 words each


def test_shared_phrases_openings_and_names_across_transcripts():
    recs = [single(f"task_{i:02d}", "s", "r") for i in range(4)]
    phrases = {p["phrase"] for p in v.shared_phrases(recs, min_share=0.5)}
    assert "clinic referral records against the" in phrases
    oc = v.openings_and_closings(recs)
    assert oc["distinct_openings"] == 1 and oc["openings"][0]["transcripts"] == 4
    names = v.cast_names(recs)
    assert names["full_names"][0] == {"name": "Ben Ode", "transcripts": 4, "share": 1.0}
    assert {"name": "Chen", "transcripts": 4, "share": 1.0} in names["surnames"]


def test_near_duplicates_only_within_a_task():
    same = [single("jargon_01", "s", "r"), single("jargon_01", "s", "r")]
    other = [single("jargon_02", "s", "r")]
    pairs = v.near_duplicates(same + other)
    assert [(p["a"], p["b"]) for p in pairs] == [(0, 1)] and pairs[0]["jaccard"] == 1.0


def test_pass_rate_and_zero_pass_cells():
    recs = [single("m_01", "screen", "r", status="failed"), single("m_01", "screen", "r", status="failed"),
            single("m_01", "quick", "r"), single("m_02", "screen", "r")]
    rates = {x["value"]: x["rate"] for x in v.pass_rate_by(recs, "style", min_n=1)}
    assert rates == {"screen": round(1 / 3, 3), "quick": 1.0}
    assert v.zero_pass_cells(recs, "style") == [{"task_id": "m_01", "value": "screen", "n": 2}]


def test_pairing_flags_setup_mismatch_between_generators():
    recs = [single("j_01", "quick", "r1", gen="gpt", seed=3), single("j_01", "quick", "r2", gen="claude", seed=3),
            single("j_01", "quick", "r1", gen="gpt", seed=4)]
    p = v.pairing(recs)
    assert p["paired_samples"] == 1
    assert p["mismatches"] == [{"task_id": "j_01", "seed": 3, "field": "research_id",
                                "values": {"gpt": "r1", "claude": "r2"}}]


def test_cli_writes_report_and_comparison(tmp_path):
    def save(root, task, status, style):
        rec = {"task_id": task, "meeting_style": style, "research_id": "r1", "generator_model": "gpt",
               "disciplines": ["a", "b"], "speakers": [{"name": "A B"}],
               "transcript": [{"speaker": "A B", "text": LONG}], "qa_feedback": {}}
        p = root / task / status / f"{task}_{status}_{style}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec))
    new, old = tmp_path / "new", tmp_path / "old"
    for s in ("quick", "tangent", "quick"):
        save(new, "jargon_01", "approved", s)
    save(old, "jargon_01", "failed", "quick")
    styles, openings, research = cli.banks()
    rep = v.report(cli.load_records([new]), styles, openings, research)
    prev = v.report(cli.load_records([old]), styles, openings, research)
    json.dumps(rep)                                   # discipline tuples serialize
    md = cli.markdown(rep, "Test", prev)
    assert "## Compared with the other batch" in md
    assert "| quick | 0% of 1 |" not in md            # min_n=3 hides small groups
    assert "| samples | 1 | 2 |" in md                # two files: quick is overwritten once
