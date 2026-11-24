"""
Offline tests for utils/beats.py: parsing, linked-beat separation and
scheduling. No API calls.

    python tests/test_beats.py
"""
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace as S

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENAI_API_KEY", "offline-test")  # client is built at import

from utils import beats  # noqa: E402

CAST = [S(name="Dr. Grace Lee", discipline="public health"),
        S(name="Dr. Ana Ruiz", discipline="environmental science"),
        S(name="Sam Thompson", discipline="sociology")]


def test_parse_keeps_cast_owners_and_orders_windows():
    raw = [
        {"id": "x3", "owner": "Thompson", "window": "late", "says": "Adopt May 16.", "linked": ["x1"]},
        {"id": "x1", "owner": "Grace Lee", "window": "early", "says": "One team per morning."},
        {"id": "x2", "owner": "Nobody Here", "window": "middle", "says": "Dropped: not cast."},
        {"id": "x4", "owner": "Dr. Ana Ruiz", "window": "soon", "says": "Bad window becomes middle."},
        {"id": "x5", "owner": "Dr. Ana Ruiz", "window": "early", "says": ""},
        "not a dict",
    ]
    got = beats.parse_beats(raw, CAST)
    assert [b["id"] for b in got] == ["b1", "b2", "b3"]
    assert [b["owner"] for b in got] == ["Dr. Grace Lee", "Dr. Ana Ruiz", "Sam Thompson"]
    assert got[1]["window"] == "middle"
    assert got[2]["linked"] == ["b1"]          # renamed along with its target


def test_parse_respects_the_limit_and_drops_forward_links():
    raw = [{"id": f"x{i}", "owner": "Ruiz", "window": "middle", "says": f"Point {i}.",
            "linked": [f"x{i + 1}"]} for i in range(1, 10)]
    got = beats.parse_beats(raw, CAST, limit=4)
    assert len(got) == 4
    assert all(b["linked"] == [] for b in got)  # a link may only point back


def test_linked_beats_get_different_owners():
    """assumptions_03 test: Lee owned both the constraint and the later
    date, and proposed a workaround for her own constraint."""
    bs = beats.parse_beats([
        {"id": "a", "owner": "Lee", "window": "early", "says": "One mobile team."},
        {"id": "b", "owner": "Lee", "window": "late", "says": "Follow-up on May 15, 2028.", "linked": ["a"]},
    ], CAST)
    bs = beats.separate_linked(bs, CAST)
    assert bs[0]["owner"] != bs[1]["owner"]


def test_schedule_keeps_gaps_windows_and_not_before():
    bs = beats.parse_beats([
        {"id": "a", "owner": "Lee", "window": "early", "says": "Constraint."},
        {"id": "b", "owner": "Ruiz", "window": "early", "says": "Something else."},
        {"id": "c", "owner": "Thompson", "window": "middle", "says": "Decision.", "linked": ["a"]},
        {"id": "d", "owner": "Ruiz", "window": "late", "says": "Late point."},
    ], CAST)
    for seed in range(50):
        slots = beats.schedule(bs, max_turns=35, not_before={"Sam Thompson": 15},
                               rng=random.Random(seed))
        turns = [slots[b["id"]] for b in bs]
        assert turns == sorted(turns)
        assert all(b - a >= beats.MIN_GAP for a, b in zip(turns, turns[1:]))
        assert slots["b3"] - slots["b1"] >= beats.LINKED_GAP
        assert slots["b3"] >= 15                      # quiet speaker stays out until late
        assert 2 <= slots["b1"] <= 8                  # early window
        assert max(turns) <= 33                       # room left to close


def test_due_waits_when_the_owner_just_spoke():
    bs = beats.parse_beats([{"id": "a", "owner": "Lee", "window": "early", "says": "X."}], CAST)
    slots = {"b1": 4}
    assert beats.due(bs, slots, set(), 3, "Sam Thompson") is None      # not yet
    assert beats.due(bs, slots, set(), 4, "Dr. Grace Lee") is None     # owner just spoke
    assert beats.due(bs, slots, set(), 5, "Sam Thompson")["id"] == "b1"
    assert beats.due(bs, slots, {"b1"}, 6, "Sam Thompson") is None     # already said


def test_task_modes():
    assert not beats.uses_beats("jargon_01") and not beats.uses_beats("coordination_05")
    assert beats.uses_beats("assumptions_03")
    assert beats.beat_limit("problem_01") == beats.ANCHOR_MAX_BEATS
    assert beats.beat_limit("assumptions_03") == beats.MAX_BEATS


def test_only_hidden_conflict_tasks_keep_links():
    """The writer marked a decision 'linked' to the objection it records
    (disagreement_02); that would hide the objection from the speakers."""
    reply = {"beats": [
        {"id": "b1", "owner": "Lee", "window": "early", "says": "Objection."},
        {"id": "b2", "owner": "Ruiz", "window": "late", "says": "Decision.", "linked": ["b1"]}],
        "conditions": []}
    real = beats.call_llm_json
    beats.call_llm_json = lambda *a, **k: reply
    try:
        for task, kept in (("disagreement_02", []), ("assumptions_03", ["b1"])):
            content = S(seed=S(task_id=task, benchmark_task="t", explicit_instruction="i", ground_truth={}))
            got, _ = beats.write_beats(content, CAST)
            assert got[1]["linked"] == kept, task
    finally:
        beats.call_llm_json = real


def test_planted_beats_and_where_they_go():
    from schema import Turn
    bs = [{"id": "b1", "owner": "A", "linked": []}, {"id": "b2", "owner": "B", "linked": ["b1"]},
          {"id": "b3", "owner": "C", "linked": []}]
    assert beats.planted_ids(bs) == {"b1"}
    long = "word " * 10
    tr = [Turn(speaker=n, text=long) for n in "ABCABCABCABCABCABCABC"]   # 21 turns
    tr[3] = Turn(speaker="A", text="mm-hm")                             # too short to carry it
    # Later piece at turn 20: A's turns 1, 7 qualify (<= 10); nearest slot 6 is turn 7.
    assert beats.plant_index(bs[0], 6, tr, 20) == 6
    # Later piece at turn 9: no A turn 10 turns earlier, so the gap is relaxed.
    assert beats.plant_index(bs[0], 3, tr, 9) == 0
    assert beats.plant_index({"owner": "Z"}, 3, tr, 20) is None


def test_leak_conditions_only_include_said_beats():
    bs = [{"id": "b1", "never": "nobody connects b1 and b3"}, {"id": "b2", "never": "x"}]
    assert beats.leak_conditions(bs, ["meeting-wide"], {"b1"}) == ["meeting-wide", "nobody connects b1 and b3"]


def test_hidden_values_are_stripped_from_what_speakers_see():
    bs = [{"id": "b1", "linked": [], "says": "Approval expected March 1, 2027."},
          {"id": "b2", "linked": ["b1"], "says": "Interviews begin February 1, 2027."},
          {"id": "b3", "linked": [], "says": "Field plan due February 15."}]
    text = ("Recruitment opens in January. Interviews start Feb. 1 to catch winter planning; "
            "the IRB decision is due 1st of March.\n- Field plan due February 15. Dr. Lee agrees.")
    kept, removed = beats.strip_cues(text, beats.hidden_cues(bs))
    assert kept == "Recruitment opens in January.\n- Field plan due February 15. Dr. Lee agrees."
    assert len(removed) == 2
    times = beats.value_patterns("saliva 30 minutes after waking and at 8:00 PM")
    assert beats.strip_cues("Samples at 8 pm and 30-min after waking. Fine.", times)[0] == "Fine."


def test_later_piece_repeating_earlier_values_is_caught():
    """assumptions_03, beats_v1: the seasonal decision restated the saliva times."""
    bs = [{"id": "b1", "linked": [], "says": "Saliva 30 minutes after waking and at 8:00 PM."},
          {"id": "b2", "linked": ["b1"], "says": "Daytime workers waking at 6:00 AM in winter."},
          {"id": "b3", "linked": ["b1"], "says": "Winter baseline, saliva at 8:00 PM each wave."}]
    assert beats.repeats(bs) == ["b3 -> b1"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
