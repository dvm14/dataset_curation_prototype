"""
critic_checks.py
Code-side checks for the critic (agents/agent5_critic.py). They need no API
call, so they also run on saved batches with `python critic.py --no-llm`.

Each check targets a problem the Week 9-10 read-through found by hand that
QA did not catch:
- doubled-word disfluency ("actually, actually") in almost every transcript
- the same opening line ("thanks for making time", "before we get into it")
- lines the generator reuses across tasks ("I lost my thread there")
- strict alternation and near-equal turn counts
- people named in the meeting who were never cast (a liaison, a manager)
- tidy endings that read out a list of next steps
- question rate (Claude speakers asked in 45% of turns vs 8% for GPT, and
  noticed the planted issue)

Every check returns a dict with the turn numbers it is based on, so a
reviewer can look them up. The checks flag; they never fail a transcript on
their own.
"""
from __future__ import annotations

import re
from collections import Counter

# "that that" and "had had" are grammatical; the rest read as an inserted tic.
_GRAMMATICAL_DOUBLES = {"that", "had", "is", "do", "the"}
_DOUBLED = re.compile(r"\b([A-Za-z']+)(?:,|\s*-|\s*\.\.\.)?\s+\1\b", re.IGNORECASE)

STOCK_OPENINGS = [
    r"thanks?(?: you)?(?:,)? (?:everyone|all|everybody)?,? ?for making (?:the )?time",
    r"before we (?:get|dive|jump) into it",
    r"can (?:you|everyone|everybody) hear me",
    r"are we all (?:here|set|good)",
]

# Lines the read-through found repeated across tasks.
KNOWN_TICS = [
    r"i lost my thread",
    r"i'?d resist",
    r"that'?s a fair point",
    r"to build on (?:that|what)",
    r"just to (?:play devil'?s advocate|push back)",
]

_TITLED_NAME = re.compile(r"\b(?:Dr|Ms|Mr|Mrs|Prof|Professor)\.?\s+([A-Z][a-z]+(?:-[A-Z][a-z]+)?)")
_COMMITMENT = re.compile(
    r"\b(?:i'?ll|i will|i can (?:take|draft|send|own)|we'?ll|by (?:monday|tuesday|wednesday|"
    r"thursday|friday|next week|end of)|next steps?|action items?|owner)\b",
    re.IGNORECASE,
)
# Shorter turns are reactions and goodbyes, not content.
SUBSTANTIVE_WORDS = 12

# People start many turns with these ("I think", "That's", "And I"); a
# repeated opener only reads as a template when it starts with something else.
_COMMON_FIRST_WORDS = {
    "i", "i'd", "i'm", "i'll", "i've", "we", "we're", "we'll", "that", "that's",
    "and", "but", "so", "yes", "yeah", "okay", "ok", "right", "well", "it", "it's",
    "the", "um", "uh", "sure", "no", "oh", "just",
}

_GOODBYE = re.compile(r"\b(?:bye|see you|talk (?:soon|later|next)|have a good|thanks,? everyone)\b",
                      re.IGNORECASE)


def _text(turn) -> str:
    return turn["text"] if isinstance(turn, dict) else turn.text


def _speaker(turn) -> str:
    return turn["speaker"] if isinstance(turn, dict) else turn.speaker


def doubled_words(transcript) -> dict:
    hits = []
    for i, turn in enumerate(transcript, start=1):
        for m in _DOUBLED.finditer(_text(turn)):
            if m.group(1).lower() not in _GRAMMATICAL_DOUBLES:
                hits.append({"turn": i, "text": m.group(0)})
    return {"count": len(hits), "hits": hits}


def stock_opening(transcript) -> dict:
    opening = " ".join(_text(t) for t in transcript[:2]).lower()
    found = [p for p in STOCK_OPENINGS if re.search(p, opening)]
    return {"flag": bool(found), "patterns": found}


def known_tics(transcript) -> dict:
    hits = []
    for i, turn in enumerate(transcript, start=1):
        for p in KNOWN_TICS:
            if re.search(p, _text(turn), re.IGNORECASE):
                hits.append({"turn": i, "pattern": p})
    return {"count": len(hits), "hits": hits}


def turn_balance(transcript) -> dict:
    """Strict rotation and equal turn counts read as scripted."""
    speakers = [_speaker(t) for t in transcript]
    counts = Counter(speakers)
    n = len(counts)
    rotation = n > 1 and len(speakers) >= 2 * n and all(
        speakers[i] == speakers[i % n] for i in range(len(speakers))
    )
    spread = max(counts.values()) - min(counts.values()) if counts else 0
    return {
        "turns_per_speaker": dict(counts),
        "strict_rotation": rotation,
        "even_counts": n > 1 and spread <= 1,
    }


def uncast_names(transcript, cast_names) -> dict:
    """
    Titled names ("Ms. Brooks") that match nobody in the cast. assumptions-03
    and assumptions-02 needed a liaison or manager who was only talked about,
    because casting makes one speaker per discipline.
    """
    cast_parts = {part.lower() for name in cast_names for part in re.split(r"[\s.]+", name) if part}
    hits = []
    for i, turn in enumerate(transcript, start=1):
        for m in _TITLED_NAME.finditer(_text(turn)):
            if m.group(1).lower() not in cast_parts:
                hits.append({"turn": i, "name": m.group(0)})
    return {"names": sorted({h["name"] for h in hits}), "hits": hits}


def tidy_ending(transcript, window: int = 4) -> dict:
    """
    The last few turns assign work and nothing is left open. A goodbye
    alone is fine (Diya's closing step adds one); a recap list is what
    the read-through flagged.
    """
    # Skip the short goodbyes the closing step appends; look at the last
    # substantive turns.
    numbered = [(i, t) for i, t in enumerate(transcript, start=1)
                if len(_text(t).split()) >= SUBSTANTIVE_WORDS]
    tail = numbered[-window:]
    commit_turns = [i for i, t in tail if _COMMITMENT.search(_text(t))]
    goodbye = any(_GOODBYE.search(_text(t)) for t in transcript[-3:])
    return {"flag": len(commit_turns) >= 3, "commitment_turns": commit_turns, "goodbye": goodbye}


def repeated_openers(transcript, words: int = 2, min_repeats: int = 3) -> dict:
    """
    Turns that start the same way ("Did the ... ever come through?" four
    turns running in one GPT coordination_01 sample).
    """
    starts = {}
    for i, t in enumerate(transcript, start=1):
        if len(_text(t).split()) < SUBSTANTIVE_WORDS:
            continue
        tokens = re.findall(r"[a-z']+", _text(t).lower().replace("’", "'"))[:words]
        if tokens and tokens[0] not in _COMMON_FIRST_WORDS:
            key = " ".join(tokens)
            starts.setdefault(key, []).append(i)
    repeated = {k: v for k, v in starts.items() if len(v) >= min_repeats}
    return {"repeated": repeated}


def question_rate(transcript) -> dict:
    turns = [i for i, t in enumerate(transcript, start=1) if "?" in _text(t)]
    rate = round(len(turns) / len(transcript), 2) if transcript else 0.0
    return {"rate": rate, "turns": turns}


def run_checks(transcript, cast_names) -> dict:
    """All code-side checks for one transcript, plus the flags they raise."""
    checks = {
        "doubled_words": doubled_words(transcript),
        "stock_opening": stock_opening(transcript),
        "known_tics": known_tics(transcript),
        "turn_balance": turn_balance(transcript),
        "uncast_names": uncast_names(transcript, cast_names),
        "tidy_ending": tidy_ending(transcript),
        "repeated_openers": repeated_openers(transcript),
        "question_rate": question_rate(transcript),
    }
    checks["flags"] = check_flags(checks)
    return checks


def check_flags(checks) -> list[str]:
    flags = []
    if checks["doubled_words"]["count"] >= 2:
        hits = checks["doubled_words"]["hits"]
        flags.append(f"Doubled-word disfluency {len(hits)} times "
                     f"(turns {', '.join(str(h['turn']) for h in hits)}).")
    if checks["stock_opening"]["flag"]:
        flags.append("Opens with a stock line the generator reuses.")
    if checks["known_tics"]["count"]:
        flags.append("Reused generator line(s) in turns "
                     + ", ".join(str(h["turn"]) for h in checks["known_tics"]["hits"]) + ".")
    balance = checks["turn_balance"]
    if balance["strict_rotation"]:
        flags.append("Speakers take turns in a fixed rotation.")
    elif balance["even_counts"]:
        flags.append(f"Turn counts are nearly equal: {balance['turns_per_speaker']}.")
    if checks["uncast_names"]["names"]:
        flags.append("Named but never cast: " + ", ".join(checks["uncast_names"]["names"]) + ".")
    if checks["tidy_ending"]["flag"]:
        flags.append("Ends on a list of assignments (turns "
                     + ", ".join(map(str, checks["tidy_ending"]["commitment_turns"])) + ").")
    for opener, turns in checks["repeated_openers"]["repeated"].items():
        flags.append(f'{len(turns)} turns open with "{opener}" (turns {", ".join(map(str, turns))}).')
    return flags
