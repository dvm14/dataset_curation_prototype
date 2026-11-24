"""
The few moments a task cannot do without, planned before the meeting and
placed in it by the code.

The meeting is improvised, but the tasks need specific moments: an early
constraint and a later decision nobody connects, one objection and then
silence, a glossary before the terms are used. Left to the improvisation,
those moments kept not happening: in a 5-runs-per-task GPT batch most
failures were the core constraint, decision or statement never said, and
re-judging with a judge told to ignore secondary detail rescued almost
none of them. Nudging speakers until they said things had already failed
(required_facts.py: meetings ran to the turn cap, and nudges spelled out
the hidden conflict).

So a beat is planned up front (one cast speaker, one point, a window of
the meeting) and scheduled to a turn. At that turn the owner takes the
floor and is asked to get the point across in passing; the turn is checked
and regenerated once if the point didn't come across. Every other turn is
improvised as before. In a test on 41 beats, all were conveyed within two
tries, and in blind pairwise comparisons beat turns were judged the more
natural of the pair about 40% of the time against a normal turn by the
same speaker at the same point.
"""
import json
import re
import random

from utils.llm import call_llm_json
from utils.names import match_speaker

MAX_BEATS = 7
# Tasks that summarise what was discussed: a few anchors, the rest is left
# to the meeting. With full beats these became a script of the whole
# meeting (8 beats in about 30 turns).
ANCHOR_TASKS = {
    "problem_01", "problem_04", "coordination_01", "coordination_02",
    "methods_01", "opportunity_01", "opportunity_02",
}
ANCHOR_MAX_BEATS = 4
# Tasks with their own mechanism that already works: the seed-term checks
# (jargon_01/02, 3/5 approved in the GPT batch) and the action-item wrap-up.
NO_BEAT_TASKS = {"jargon_01", "jargon_02", "coordination_05"}
# Tasks whose meeting must contain two things nobody connects (a constraint
# and a later decision, a date and a dependency, a question and a method
# that can't answer it). Only these get linked beats, and with them agent3's
# memory window. Left to the writer, "linked" was set for 11 of 20 tasks,
# e.g. a decision "linked" to the objection it has to record.
HIDDEN_LINK_TASKS = {"assumptions_03", "coordination_03", "methods_03"}

WINDOWS = {"early": (2, 8), "middle": (9, 18), "late": (19, 28)}
MIN_GAP = 3        # turns between any two beats
LINKED_GAP = 10    # turns between the pieces of a hidden conflict
BEAT_RETRIES = 1

BEAT_PROMPT = """
You plan the few key moments a simulated research meeting must contain so
that a benchmark task can be tested on its transcript. The rest of the
meeting is improvised by the participants; you only fix what the task
cannot do without.

Return at most {limit} beats, and fewer is better: if the task depends on
three moments, return three. Each beat is one cast speaker bringing up ONE
point that fits in a sentence or two of normal speech:
- "id": "b1", "b2", ... in meeting order
- "owner": the exact name of a cast speaker
- "window": "early" | "middle" | "late"
- "says": what the owner brings up, as content in plain words (not a line
  of dialogue), at most 25 words. Include exact values (dates, numbers,
  names, thresholds) when the task depends on them. If something needs
  more than 25 words, it is several points: split it into separate beats,
  given to different speakers or spread over the meeting.
- "linked": only when hidden_conflict is true: ids of earlier beats this
  one must stay unconnected to (the pieces of the conflict the evaluated AI
  has to find). Otherwise always []
- "never": optional, what must not happen in this turn or afterwards

Rules:
- Only what the evaluated AI actually needs to answer the task. Leave out
  background detail, extra sub-items and colour; the meeting adds its own.
  A plan needs who does what and what depends on what, not every
  deliverable's full contents.
- No "discussion topic" beats ("X leads discussion of A, B and C"): the
  improvised turns cover ordinary discussion on their own.
- Owners must be cast speakers. If the seed names a role or person who is
  not cast, give that part to the cast speaker whose discipline fits, and
  don't mention that person anywhere in a beat ("says" included).
- A decision is a beat too: the owner proposes it and the group settles on
  it ("the group adopts X").
- Hidden issues (a conflict, mismatch, gap or ambiguity the AI must find):
  give the pieces to different owners and windows, list the earlier piece
  in "linked", and never write a beat that states the link, the conflict,
  the mismatch or the answer. Each piece states only its own values: a
  later piece never repeats a date, time or number from the piece it is
  linked to.
- Controls (the target problem must be absent): beats establish what must
  be present (e.g. definitions early), and "never" holds the problem that
  must not appear.
- Things that must stay unresolved, unassigned or unanswered: say so in
  "never".
- Don't make anyone perform the evaluated AI's analysis for it.

Also return "conditions": meeting-wide requirements that are not a single
moment (e.g. "nobody notices that b1 and b4 conflict"). Keep it short.

Return JSON: {{"beats": [...], "conditions": [...]}}
"""

CHECK_PROMPT = """
You check one new turn of a meeting against a point the speaker was meant
to get across. Paraphrase is fine; exact values (dates, numbers, names)
must be right. A decision counts if the turn proposes or confirms it.
Return JSON {"conveyed": true | false, "missing": "what is not conveyed, or empty"}
"""

PLANT_PROMPT = """
You edit one turn of a meeting transcript. Add the given point to it as a
brief aside in the speaker's own voice, where it fits naturally (often at
the end: "oh, and ..."). Keep everything else in the turn as it is. State
the point plainly, with its exact values; don't explain its consequences or
connect it to anything else in the meeting.
Return JSON {"text": "the edited turn"}
"""

GROUND_TRUTH_PROMPT = """
You write the reference answer for a benchmark item from a finished meeting
transcript. The evaluated AI will be scored against it, so it must match what
the transcript actually says.

You get: benchmark_task; draft_ground_truth, written before the meeting (the
intended answer and its format); planned_moments, with the turn each was
placed at; conditions; and the numbered transcript.

Write the ground truth for this transcript:
- Keep the draft's structure and keys where they still fit.
- Change names, roles, values and wording to match the transcript (if a
  different person proposed something, say so).
- Include only what the transcript supports, and cite turn numbers.
- Leave out anything in the draft that is not in the transcript, and list
  it under "not_in_transcript".
- Don't add analysis beyond what the task asks the evaluated AI for.
- For a control, the answer states that the target problem is absent.

Return JSON {"ground_truth": {...}, "changes": ["what differs from the draft, and why", ...]}
"""

BEAT_NOTE = (
    "In this turn, bring up the following in your own words, as one point "
    "among what you'd naturally say right now. Say the part that matters in "
    "passing; don't announce it, explain it or justify it:\n{says}"
)


def uses_beats(task_id):
    return task_id not in NO_BEAT_TASKS


def beat_limit(task_id):
    return ANCHOR_MAX_BEATS if task_id in ANCHOR_TASKS else MAX_BEATS


def write_beats(content, speakers):
    """Plan the task's beats with the cast. Returns (beats, conditions)."""
    limit = beat_limit(content.seed.task_id)
    setup = {
        "benchmark_task": content.seed.benchmark_task,
        "seed_instruction": content.seed.explicit_instruction,
        "ground_truth": content.seed.ground_truth,
        "cast": [{"name": s.name, "discipline": s.discipline} for s in speakers],
        "hidden_conflict": content.seed.task_id in HIDDEN_LINK_TASKS,
    }
    hidden_task = content.seed.task_id in HIDDEN_LINK_TASKS
    for attempt in range(2):
        raw = call_llm_json(BEAT_PROMPT.format(limit=limit),
                            json.dumps(setup, indent=2), max_tokens=2500)
        beats = parse_beats(raw.get("beats"), speakers, limit)
        if not hidden_task:
            for b in beats:
                b["linked"] = []
            break
        # A later piece that restates the earlier one's values (assumptions_03:
        # the seasonal decision repeated the fixed saliva times) puts the
        # conflict in a single turn.
        repeated = repeats(beats)
        if not repeated:
            break
        print(f"      Beats: a later piece repeats an earlier one ({repeated})"
              + (", re-planning" if attempt == 0 else ", kept"))
        setup["fix"] = ("A linked beat repeated values from the earlier beat it is "
                        f"linked to ({repeated}). Each piece states only its own values.")
    beats = separate_linked(beats, speakers)
    conditions = [str(c).strip() for c in raw.get("conditions") or [] if str(c).strip()]
    return beats, conditions


def repeats(beats):
    """'later -> earlier' for each linked beat that states one of the
    exact values of an earlier beat it is linked to."""
    by_id = {b["id"]: b for b in beats}
    return [f"{b['id']} -> {e}" for b in beats for e in b["linked"] if e in by_id
            and any(re.search(p, b["says"], re.I) for p in value_patterns(by_id[e]["says"]))]


def parse_beats(raw, speakers, limit=MAX_BEATS):
    """
    Keep well-formed beats with a cast owner, in window order, renumbered
    b1..bn. "linked" keeps only earlier beats that survived.
    """
    beats = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        owner = match_speaker(str(item.get("owner", "")), speakers)
        says = str(item.get("says", "")).strip()
        if not owner or not says:
            continue
        window = item.get("window") if item.get("window") in WINDOWS else "middle"
        linked = item.get("linked") if isinstance(item.get("linked"), list) else []
        beats.append({"old_id": str(item.get("id", "")), "owner": owner, "window": window,
                      "says": says, "never": str(item.get("never") or "").strip(),
                      "linked": [str(x) for x in linked]})
    order = list(WINDOWS)
    beats.sort(key=lambda b: order.index(b["window"]))     # stable: keeps the writer's order within a window
    beats = beats[:limit]
    position = {}
    for i, b in enumerate(beats, start=1):
        position[b["old_id"]] = i
        b["id"] = f"b{i}"
    for i, b in enumerate(beats, start=1):
        b["linked"] = [f"b{position[x]}" for x in b["linked"]
                       if x in position and position[x] < i]
        del b["old_id"]
    return beats


def separate_linked(beats, speakers):
    """
    Pieces of a hidden issue must come from different people: when one
    speaker owned both the constraint and the later decision, they connected
    them and proposed a workaround. Reassign the later beat if needed.
    """
    owner_of = {b["id"]: b["owner"] for b in beats}
    for b in beats:
        taken = {owner_of[x] for x in b["linked"]}
        if b["owner"] in taken:
            others = [s.name for s in speakers if s.name not in taken]
            if others:
                b["owner"] = others[0]
                owner_of[b["id"]] = b["owner"]
    return beats


def schedule(beats, max_turns, not_before=None, rng=random):
    """
    Turn number (1-based) for each beat: spread evenly over its window (a
    random pick per beat let one late pick push every later beat to the end
    of the meeting), at least MIN_GAP after the previous beat and LINKED_GAP
    after a beat it is linked to, and not before not_before[owner] (e.g. a
    quiet speaker who stays out until late). Returns {beat id: turn}.
    """
    not_before = not_before or {}
    last_turn = max_turns - 2               # leave room to close
    per_window = {w: [b for b in beats if b["window"] == w] for w in WINDOWS}
    slots, prev = {}, 0
    for b in beats:
        lo, hi = WINDOWS[b["window"]]
        group = per_window[b["window"]]
        j, k, width = group.index(b), len(group), hi - lo + 1
        sub_lo, sub_hi = lo + width * j // k, lo + width * (j + 1) // k - 1
        lo = max([sub_lo, prev + MIN_GAP, not_before.get(b["owner"], 0)]
                 + [slots[x] + LINKED_GAP for x in b["linked"] if x in slots])
        lo = min(lo, last_turn)
        hi = min(max(sub_hi, lo), last_turn)
        slots[b["id"]] = prev = rng.randint(lo, hi)
    return slots


def due(beats, slots, done, turn_number, last_speaker):
    """
    The first unsaid beat whose turn has come. If its owner just spoke
    (no back-to-back turns), it waits one turn.
    """
    for b in beats:
        if b["id"] not in done and slots[b["id"]] <= turn_number:
            return None if b["owner"] == last_speaker else b
    return None


def planted_ids(beats):
    """
    The earlier pieces of a hidden conflict. They are not spoken live: once
    a constraint had been said, everyone who heard it connected it to the
    later decision, whatever they were told or shown (two test runs, 37 leak
    regenerations). They are added afterwards as an aside in one of the
    owner's early turns, so no later turn was written knowing them.
    """
    return {x for b in beats for x in b["linked"]}


def plant_index(beat, slot, transcript, before_turn):
    """
    Index of the owner's turn to carry a planted beat: a substantive turn
    (8+ words) at least LINKED_GAP turns before before_turn (the later
    piece), nearest the scheduled slot. Relaxes the gap, then returns None
    if the owner has no turn before the later piece at all.
    """
    owned = [i for i, t in enumerate(transcript)
             if t.speaker == beat["owner"] and len(t.text.split()) >= 8]
    for limit in (before_turn - LINKED_GAP, before_turn - 1):
        options = [i for i in owned if i + 1 <= limit]
        if options:
            return min(options, key=lambda i: abs(i + 1 - slot))
    return None


def plant(beat, turn_text):
    """The turn rewritten with the beat added as an aside."""
    raw = call_llm_json(PLANT_PROMPT, json.dumps(
        {"point": beat["says"], "turn": turn_text}, indent=2), max_tokens=600)
    return str(raw.get("text") or "").strip()


def write_ground_truth(content, transcript, beats, conditions):
    """
    Reference answer written from the finished transcript. Agent 1's draft
    was written before the meeting, so it named people who were never cast
    and gave parts to whoever the seed named, while beats give them to cast
    speakers. The draft stays in seed_spec. Returns (ground_truth, changes);
    the draft, unchanged, if the reply is unusable.
    """
    raw = call_llm_json(GROUND_TRUTH_PROMPT, json.dumps({
        "benchmark_task": content.seed.benchmark_task,
        "draft_ground_truth": content.seed.ground_truth,
        "planned_moments": [{k: b.get(k) for k in ("id", "owner", "says", "turn")} for b in beats],
        "conditions": conditions,
        "transcript": [f"[{i}] {t.speaker}: {t.text}" for i, t in enumerate(transcript, start=1)],
    }, indent=2), max_tokens=4000)
    truth = raw.get("ground_truth")
    if not isinstance(truth, dict) or not truth:
        return content.seed.ground_truth, ["Kept the draft: no usable ground truth returned."]
    return truth, [str(c) for c in raw.get("changes") or []]


def beat_note(beat):
    return BEAT_NOTE.format(says=beat["says"])


def check_beat(beat, recent_turns, text):
    """(conveyed, missing) for a turn that was meant to carry the beat."""
    raw = call_llm_json(CHECK_PROMPT, json.dumps({
        "point": beat["says"], "recent_turns": recent_turns, "new_turn": text,
    }, indent=2), max_tokens=200)
    return raw.get("conveyed") is True, str(raw.get("missing") or "")


def leak_conditions(beats, conditions, done):
    """What later turns must not do: meeting-wide conditions plus the
    'never' of every beat already said."""
    return list(conditions) + [b["never"] for b in beats if b["id"] in done and b["never"]]


# ── keeping a hidden conflict's pieces out of what speakers are told ─────

_MONTHS = ["january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december"]
_MONTH = r"(" + "|".join(m[:3] + r"[a-z]*" for m in _MONTHS) + r")\.?"
_UNITS = (r"(minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?|meters?|metres?|"
          r"km|kilometers?|kilometres?|miles?|percent|%)")


def _month_re(name):
    m = _MONTHS[[x[:3] for x in _MONTHS].index(name[:3].lower())]
    return m[:3] + (r"[a-z]*" if m != "may" else "")


def value_patterns(text):
    """Regexes for the exact values in a beat: dates (month + day, either
    order), clock times and numbers with a unit. Each matches the common
    ways of saying the same value ("Feb 1", "1 February", "8 p.m.")."""
    found = []
    for m in re.finditer(r"\b" + _MONTH + r"\s+(\d{1,2})(?:st|nd|rd|th)?\b", text, re.I):
        mon, day = _month_re(m.group(1)), m.group(2)
        found.append(rf"\b(?:{mon}\.?\s+{day}(?:st|nd|rd|th)?\b|{day}(?:st|nd|rd|th)?\s+(?:of\s+)?{mon}\b)")
    for m in re.finditer(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + _MONTH, text, re.I):
        mon, day = _month_re(m.group(2)), m.group(1)
        found.append(rf"\b(?:{mon}\.?\s+{day}(?:st|nd|rd|th)?\b|{day}(?:st|nd|rd|th)?\s+(?:of\s+)?{mon}\b)")
    for m in re.finditer(r"\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s?m\b\.?", text, re.I):
        h, mm, ap = m.groups()
        minutes = rf"(?::{mm})?" if mm in (None, "00") else rf":{mm}"
        found.append(rf"\b{h}{minutes}\s*{ap}\.?\s?m\b")
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*-?\s*" + _UNITS + r"(?=\W|$)", text, re.I):
        unit = m.group(2).lower()
        stem = "%" if unit == "%" else re.escape(unit[:3] if len(unit) > 3 else unit)
        found.append(rf"\b{re.escape(m.group(1))}\s*-?\s*{stem}")
    return list(dict.fromkeys(found))


def hidden_cues(beats):
    """Value patterns from the hidden pieces of a conflict (planted or
    linked beats) that no other beat also needs."""
    hidden = planted_ids(beats) | {b["id"] for b in beats if b["linked"]}
    shown = " ".join(b["says"] for b in beats if b["id"] not in hidden)
    cues = [p for b in beats if b["id"] in hidden for p in value_patterns(b["says"])]
    return [p for p in dict.fromkeys(cues) if not re.search(p, shown, re.I)]


def strip_cues(text, cues):
    """(text without the sentences that contain a cue, removed sentences)."""
    if not cues or not text:
        return text, []
    pattern = re.compile("|".join(cues), re.I)
    removed, lines = [], []
    for line in text.split("\n"):
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])|(?<=;)\s+", line)
        kept = [s for s in parts if not pattern.search(s)]
        removed += [s.strip() for s in parts if pattern.search(s)]
        if kept != parts and not "".join(kept).strip(" -•*"):
            continue                      # the whole line (e.g. a bullet) went
        lines.append(" ".join(kept))
    return "\n".join(lines), removed
