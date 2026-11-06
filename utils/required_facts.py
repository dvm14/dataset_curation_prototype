"""
required_facts.py
Make sure the facts a task depends on are actually said in the meeting.

The planner puts each required fact (a date, a number, an owner, a
definition) in one speaker's private brief, but nothing made that speaker
say it: in a coordination_03 run nobody ever said that interviews start
on February 1, 2027, so the conflict the task is about wasn't there.

The planner now also returns each fact with a short exact cue ("February
1, 2027"). A fact counts as said once one turn contains every word of its
cue; a "1" also matches "1st". Speakers whose facts are still unsaid get
a private nudge, and the meeting can't end until every fact is said
for a while (FACT_GATE_TURNS). This is the general version of the seed-term
nudge used for terminology tasks.
"""

import re

MAX_FACTS = 8
# The meeting can't end with facts unsaid until this turn; after that it
# may. Blocking until MAX_TURNS made 10 of 11 meetings run to the 35-turn
# cap, and naturalness dropped from 3 to 2 in most of them.
FACT_GATE_TURNS = 20


def _cue_words(cue):
    return [w for w in re.findall(r"[a-z0-9]+", cue.lower()) if w]


def _has_word(text, word):
    suffix = r"(st|nd|rd|th)?" if word.isdigit() else r"(s|es)?"
    return re.search(rf"\b{re.escape(word)}{suffix}\b", text, re.IGNORECASE) is not None


def cue_said(text, cue):
    """
    All cue words appear in the turn. A year is optional when the cue has
    other words: people say "March 18th", not "March 18, 2027", and
    requiring the year kept speakers nudged for the rest of the meeting.
    """
    words = _cue_words(cue)
    required = [w for w in words if not re.fullmatch(r"(19|20)\d\d", w)] or words
    return bool(required) and all(_has_word(text, w) for w in required)


def is_value_cue(cue):
    """
    Only track cues that name a value: something with a digit (a date,
    number, threshold) or a capitalized name ("Priya", "WHO-5"). The
    planner also returned phrases like "cannot occur until approved" or
    "as part of those interviews": they never match verbatim, so the
    speaker was nudged all meeting, and some spelled out the very
    conflict the task needs left unnoticed.
    """
    return bool(re.search(r"\d", cue)) or any(w[:1].isupper() for w in cue.split())


def parse_facts(raw, speakers):
    """
    Keep well-formed facts assigned to a real speaker, with a value cue
    short enough to match (at most 6 words). Anything else is dropped
    rather than tracked badly.
    """
    names = {s.name for s in speakers}
    facts = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        speaker, fact, cue = (str(item.get(k, "")).strip() for k in ("speaker", "fact", "cue"))
        if speaker in names and fact and 0 < len(_cue_words(cue)) <= 6 and is_value_cue(cue):
            facts.append({"speaker": speaker, "fact": fact, "cue": cue})
    return facts[:MAX_FACTS]


def unsaid(facts, transcript):
    """Facts whose cue no turn has said yet (by anyone)."""
    return [f for f in facts if not any(cue_said(t.text, f["cue"]) for t in transcript)]


def fact_nudge(facts):
    items = "\n".join(f"- {f['fact']}" for f in facts)
    return (
        "Something from your brief you haven't mentioned yet:\n"
        f"{items}\n"
        "If it fits the conversation, bring it up now as you normally "
        "would, giving the specifics (exact dates and numbers)."
    )


def cues_in(text, facts):
    return {f["cue"] for f in facts if cue_said(text, f["cue"])}
