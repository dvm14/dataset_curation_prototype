"""
Task constraints that the code enforces instead of asking the speakers.

The Claude pilot showed speakers "fixing" what a task needs left in place:
in integration_01 the dominant discipline got 50% of the words instead of
60-70% (strict turn alternation caps any one person at half the turns),
and in coordination_05 the group added a seventh action item when the
task needs exactly six. Both are countable, so they are controlled here.

Airtime (AIRTIME_TASKS): the dominant speaker takes the floor whenever
they are below their target share of words and allowed to speak; the
other speakers are kept to short turns; the quiet speaker stays out
until late, then gets the floor once for their point, and at most
QUIET_MAX_TURNS turns in all.

Action items (ground truth with assigned_items / unassigned_items): the
chair runs through exactly those items before the meeting closes, checked
in code (owners and dates said), and nobody reads out a list or adds a
to-do before then. The wrap-up's names and dates are protected from the
speech pass.
"""
import re

from utils.llm import call_llm_json
from utils.names import match_speaker

AIRTIME_TASKS = {"integration_01": 0.65}  # task -> target word share of the dominant speaker
QUIET_FROM_TURN = 15        # the quiet speaker stays out before this turn ("late in the meeting")
QUIET_MAX_TURNS = 3         # "only 2 or 3 short turns"
SHORT_TURN_WORDS = 35       # word budget for the other speakers while airtime is steered
WRAP_UP_WORDS = 170
WRAP_UP_RETRIES = 2

ROLES_PROMPT = """You read a meeting specification. Name the discipline of the participant
who should dominate the talking, and the discipline of the participant who
should speak least. Use the discipline names exactly as listed.
Return JSON: {"dominant": "<discipline>", "quiet": "<discipline>"}"""


# ── airtime ──────────────────────────────────────────────────────────────

def airtime_roles(content, speakers):
    """{'dominant': name, 'quiet': name, 'share': float} for airtime tasks, else None."""
    share = AIRTIME_TASKS.get(content.seed.task_id)
    if share is None:
        return None
    reply = call_llm_json(ROLES_PROMPT, (
        f"Disciplines: {[s.discipline for s in speakers]}\n\n"
        f"Specification:\n{content.seed.explicit_instruction}"), max_tokens=200)

    def by_discipline(value):
        value = str(value or "").strip().lower()
        hits = [s.name for s in speakers
                if value and (value in s.discipline.lower() or s.discipline.lower() in value)]
        return hits[0] if len(hits) == 1 else match_speaker(value, speakers)

    dominant, quiet = by_discipline(reply.get("dominant")), by_discipline(reply.get("quiet"))
    if not dominant or dominant == quiet:
        print("      Airtime roles unclear, leaving turn-taking alone")
        return None
    return {"dominant": dominant, "quiet": quiet, "share": share}


def word_share(transcript, name):
    words = [len(t.text.split()) for t in transcript]
    total = sum(words)
    return sum(w for t, w in zip(transcript, words) if t.speaker == name) / total if total else 0.0


def steer(roles, transcript, turn_index):
    """
    (forced speaker or None, speakers barred from the floor this turn).
    The caller still respects the no-back-to-back rule for the forced speaker.
    """
    if not roles:
        return None, set()
    quiet, dominant = roles["quiet"], roles["dominant"]
    quiet_turns = sum(t.speaker == quiet for t in transcript)
    barred = set()
    if quiet and (turn_index < QUIET_FROM_TURN or quiet_turns >= QUIET_MAX_TURNS):
        barred.add(quiet)
    if quiet and quiet_turns == 0 and turn_index >= QUIET_FROM_TURN:
        return quiet, barred                         # their one substantive point
    if word_share(transcript, dominant) < roles["share"]:
        return dominant, barred
    return None, barred


def quiet_still_due(roles, transcript):
    """The meeting can't end before the quiet speaker has made their point."""
    return bool(roles and roles["quiet"]
                and not any(t.speaker == roles["quiet"] for t in transcript))


def turn_limit(roles, name, transcript):
    """Word budget override for this speaker's turn, or None."""
    if not roles or name == roles["dominant"]:
        return None
    if name == roles["quiet"] and not any(t.speaker == name for t in transcript):
        return None                                  # the point itself gets a normal turn
    return SHORT_TURN_WORDS


# ── action items ─────────────────────────────────────────────────────────

# Tasks where a single "action_items" list (owner / status per item) is also
# read; kept to the action-item task so no other ground truth starts a wrap-up.
COMBINED_LIST_TASKS = {"coordination_05"}

def _first(item, *keys):
    for k in keys:
        if item.get(k) not in (None, ""):
            return str(item[k])
    return ""


def action_items(content):
    """(assigned, unassigned) from the ground truth, or None if the task has no item list."""
    truth = content.seed.ground_truth or {}
    assigned, unassigned = truth.get("assigned_items"), truth.get("unassigned_items")
    combined = truth.get("action_items")
    if (assigned is None and unassigned is None and isinstance(combined, list)
            and content.seed.task_id in COMBINED_LIST_TASKS):
        # Agent 1 sometimes writes one list with an owner / status per item
        items = [i for i in combined if isinstance(i, dict)]
        is_open = lambda it: (str(it.get("status", "")).lower() == "unassigned"
                              or not _first(it, "owner", "assignee"))
        assigned = [i for i in items if not is_open(i)]
        unassigned = [i for i in items if is_open(i)]
    if not isinstance(assigned, list) or not isinstance(unassigned, list) or not assigned + unassigned:
        return None
    norm = lambda it: {"item": _first(it, "item", "task", "description", "action"),
                       "owner": _first(it, "owner", "assignee"),
                       "date": _first(it, "due_date", "date", "due")}
    return [norm(i) for i in assigned if isinstance(i, dict)], [norm(i) for i in unassigned if isinstance(i, dict)]


def items_hold_note(items):
    if not items:
        return ""
    return ("Don't take on or suggest new tasks or to-dos, and don't read out a list of "
            "action items: those will be gone through once at the end.")


def wrap_up_note(items):
    assigned, unassigned = items
    lines = [f"- {i['item']} — {i['owner']}, by {i['date']}" for i in assigned]
    lines += [f"- {i['item']} — nobody has taken this yet" for i in unassigned]
    n = len(lines)
    return (
        f"Before closing, run through the action items out loud, the way a chair does: "
        f"exactly these {n}, no others, and say there are {n}.\n" + "\n".join(lines) + "\n"
        "Say each owner's name and date plainly. For the items nobody has taken, say "
        "that someone still needs to pick them up, without suggesting who. "
        "Don't add, merge or split items, and don't say goodbye yet."
    )


def owner_cue(owner, speakers):
    """The first name the chair would say ('Priya' for 'Dr. Priya Nair')."""
    name = match_speaker(owner, speakers) or owner
    words = [w for w in re.findall(r"[A-Za-z'-]+", name) if w.lower().rstrip(".") not in
             {"dr", "prof", "professor", "mr", "mrs", "ms", "mx"}]
    return words[0] if words else owner


def wrap_up_cues(items, speakers):
    """Cues the wrap-up must contain, as required-fact dicts (for the speech-pass guard)."""
    assigned, _ = items
    cues = []
    for i in assigned:
        if i["owner"]:
            cues.append({"speaker": "", "fact": i["item"], "cue": owner_cue(i["owner"], speakers)})
        if i["date"] and re.search(r"\d", i["date"]):
            cues.append({"speaker": "", "fact": i["item"], "cue": i["date"]})
    return cues


def wrap_up_missing(text, cues):
    from utils.required_facts import cue_said
    return [c["cue"] for c in cues if not cue_said(text, c["cue"])]
