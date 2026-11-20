"""The sample seed makes background/style picks repeatable across generator models."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import reseed
from utils.meeting_styles import pick_style, pick_opening
from utils.research_banks import pick_research


def picks(task_id, seed, noise=0):
    config = {"task_id": task_id, "sample_seed": seed}
    reseed(config, "style")
    style = (pick_style()[0], pick_opening())
    for _ in range(noise):          # e.g. an API retry drawing random numbers
        random.random()
    reseed(config, "research")
    return style, pick_research(None, 3, "distant")["id"]


def test_same_seed_same_picks_even_after_extra_draws():
    assert picks("jargon_01", 3) == picks("jargon_01", 3, noise=5)


def test_seeds_and_tasks_vary():
    assert len({picks("jargon_01", k) for k in range(1, 11)}) > 3
    assert picks("jargon_01", 1) != picks("problem_01", 1) or picks("jargon_01", 2) != picks("problem_01", 2)


def test_no_seed_leaves_random_alone():
    random.seed(0)
    before = random.random()
    random.seed(0)
    reseed({"task_id": "jargon_01", "sample_seed": None}, "style")
    assert random.random() == before
