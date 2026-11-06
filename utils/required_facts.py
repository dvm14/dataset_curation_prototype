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
(MAX_TURNS still caps it). This is the general version of the seed-term
nudge used for terminology tasks.
"""

import re

MAX_FACTS = 8


def _cue_words(cue):
    return [w for w in re.findall(r"[a-z0-9]+", cue.lower()) if w]


def _has_word(text, word):
    suffix = r"(st|nd|rd|th)?" if word.isdigit() else r"(s|es)?"
    return re.search(rf"\b{re.escape(word)}{suffix}\b", text, re.IGNORECASE) is not None


def cue_said(text, cue):
    words = _cue_words(cue)
    return bool(words) and all(_has_word(text, w) for w in words)


def parse_facts(raw, speakers):
    """
    Keep well-formed facts assigned to a real speaker, with a cue short
    enough to match (at most 6 words). Anything else is dropped rather
    than tracked badly.
    """
    names = {s.name for s in speakers}
    facts = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        speaker, fact, cue = (str(item.get(k, "")).strip() for k in ("speaker", "fact", "cue"))
        if speaker in names and fact and 0 < len(_cue_words(cue)) <= 6:
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
