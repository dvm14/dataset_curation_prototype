"""Offline checks for the research and meeting-style banks (no API calls)."""
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.research_banks import RESEARCH_BANK, pick_research, research_text
from utils.meeting_styles import (MEETING_STYLES, MEETING_OPENINGS,
                                  STOCK_PHRASES_TO_AVOID, pick_style, style_block)


def test_entries_are_complete():
    ids = [e["id"] for e in RESEARCH_BANK]
    assert len(ids) == len(set(ids))
    for e in RESEARCH_BANK:
        assert e["distance"] in {"close", "distant", "mixed"}
        assert len(e["disciplines"]) >= 2
        assert set(e["lenses"]) == set(e["disciplines"]), e["id"]


def test_default_run_gets_variety():
    random.seed(1)
    picks = [pick_research(None, 3, "distant") for _ in range(400)]
    assert all(len(e["disciplines"]) == 3 for e in picks)
    assert all(set(e["lenses"]) == set(e["disciplines"]) for e in picks)
    # three-discipline entries plus the grown two-discipline ones
    assert len({e["id"] for e in picks}) >= 18


def test_two_discipline_entries_grow_to_three_without_changing_the_bank():
    two = [e for e in RESEARCH_BANK if len(e["disciplines"]) == 2]
    assert two and all("third" in e for e in two)
    random.seed(2)
    grown = [e for e in (pick_research(None, 3) for _ in range(400))
             if e["id"] in {t["id"] for t in two}]
    assert grown
    for e in grown:
        base = next(t for t in two if t["id"] == e["id"])
        assert e["disciplines"][:2] == base["disciplines"]
        assert e["disciplines"][2] == base["third"]["discipline"]
        assert e["disciplines"][2] not in base["disciplines"]
        assert len(base["disciplines"]) == 2  # bank itself is untouched


def test_two_speaker_runs_use_two_discipline_entries_as_they_are():
    random.seed(3)
    picks = [pick_research(None, 2, "distant") for _ in range(100)]
    assert all(len(e["disciplines"]) == 2 for e in picks)


def test_task_discipline_override_is_matched():
    e = pick_research(["clinical medicine", "sociology"])
    assert set(e["disciplines"]) == {"clinical medicine", "sociology"}
    text = research_text(e, only=["clinical medicine", "sociology"])
    assert e["problem"] in text and "anthropology:" not in text


def test_unknown_disciplines_return_none():
    assert pick_research(["astronomy", "law"]) is None


def test_lenses_limited_to_cast_disciplines():
    e = pick_research(["ecology", "economics"])
    assert e is not None
    text = research_text(e, only=["ecology", "economics"])
    assert "anthropology:" not in text


def test_style_block_bans_stock_phrases_and_validates_names():
    name, style = pick_style()
    block = style_block(name, style)
    for phrase in STOCK_PHRASES_TO_AVOID:
        assert phrase in block
    assert any(o in block for o in MEETING_OPENINGS)
    try:
        pick_style("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown style should raise")


def test_every_style_has_description_and_examples():
    for name, s in MEETING_STYLES.items():
        assert s["description"] and len(s["examples"]) >= 2, name


def test_seniority_task_never_gets_seniority_styles():
    from utils.meeting_styles import pick_style
    flagged = {n for n, st in MEETING_STYLES.items() if st.get("shows_seniority")}
    assert flagged == {"deferential_junior", "one_voice_leads"}
    random.seed(0)
    for _ in range(300):
        name, _style = pick_style(avoid_seniority=True)
        assert name not in flagged
    for name in flagged:
        try:
            pick_style(name, avoid_seniority=True)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{name} should be refused for seniority tasks")
    # other tasks can still use them
    assert pick_style("deferential_junior")[0] == "deferential_junior"


def test_assumptions_02_pair_has_several_backgrounds():
    ids = {pick_research(["clinical medicine", "sociology"])["id"] for _ in range(200)}
    assert len(ids) >= 3


def test_saved_output_records_style_opening_and_research_id():
    """Single-meeting and series outputs carry the bank choices as fields."""
    import tests.test_task_pipeline as single
    import tests.test_multi_meeting as multi
    from utils.research_banks import RESEARCH_BANK

    result, _ = single.run_task("jargon_01")
    ids = {e["id"] for e in RESEARCH_BANK}
    assert result.research_id in ids
    assert result.meeting_style in MEETING_STYLES
    assert result.meeting_opening in MEETING_OPENINGS
    assert result.model_dump()["meeting_style"] == result.meeting_style

    out, _ = multi.run_series("self_review_01")
    assert out["research_id"] in ids
    assert out["meeting_style"] in MEETING_STYLES
    assert out["meeting_opening"] in MEETING_OPENINGS


def test_seniority_pair_run_never_uses_seniority_styles(monkeypatch):
    # Call the real style step many times with different seeds. (The series
    # test helper re-seeds the RNG, so it cannot be used for this.)
    import main
    import utils.style_extractor as style_extractor
    monkeypatch.setattr(style_extractor, "call_llm", lambda *a, **k: "notes")
    flagged = {"deferential_junior", "one_voice_leads"}
    seen = set()
    for seed in range(200):
        random.seed(seed)
        config = {**main.CONFIG, "task_id": "other_02"}
        main.load_ami_style(config)
        seen.add(config["meeting_style_used"])
    assert not (seen & flagged)
    assert len(seen) >= 5  # the other styles still get used

    # a normal task can still draw them
    seen = set()
    for seed in range(200):
        random.seed(seed)
        config = {**main.CONFIG, "task_id": "jargon_01"}
        main.load_ami_style(config)
        seen.add(config["meeting_style_used"])
    assert flagged <= seen


def test_task_style_exclusions():
    from utils.meeting_styles import TASK_STYLE_EXCLUDE
    for task_id, banned in TASK_STYLE_EXCLUDE.items():
        assert set(banned) <= set(MEETING_STYLES), task_id
        picked = {pick_style(task_id=task_id)[0] for _ in range(500)}
        assert not picked & set(banned), task_id
        try:
            pick_style(banned[0], task_id=task_id)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{banned[0]} should be refused for {task_id}")
    assert "quick_checkin" in {pick_style(task_id="jargon_01")[0] for _ in range(500)}


def test_screen_share_says_values_aloud():
    assert "out loud" in MEETING_STYLES["screen_share_working_session"]["description"]
