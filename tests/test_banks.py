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
    seen = Counter(pick_research(None, 3, "distant")["id"] for _ in range(200))
    assert len(seen) >= 4
    assert all(len(next(e for e in RESEARCH_BANK if e["id"] == i)["disciplines"]) == 3
               for i in seen)


def test_task_discipline_override_is_matched():
    e = pick_research(["clinical medicine", "sociology"])
    assert set(e["disciplines"]) == {"clinical medicine", "sociology"}
    text = research_text(e, only=["clinical medicine", "sociology"])
    assert "readmissions" in text


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
