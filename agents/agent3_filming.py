"""
agent3_filming.py
Filming Agent — simulates the meeting. Each cast speaker is a
separate agent with a private brief. Nobody assigns turns: every
round, each agent bids for the floor and the strongest bid speaks.
"""

import json
import random
from concurrent.futures import ThreadPoolExecutor

from schema import (
    FilmingOutput,
    Turn,
    ContentCurationOutput,
    CastingOutput,
)
from utils.llm import call_llm_json, CACHE_BREAKPOINT
from prompts.agent3_prompt import (
    SYSTEM_PROMPT, BID_PROMPT, LEAK_CHECK_PROMPT, SPEECH_PASS_PROMPT,
)


# One turn means one participant speaks once.
MIN_TURNS = 15          # the meeting can't end on its own before this
MAX_TURNS = 35          # hard stop
MAX_CONSECUTIVE = 1     # no back-to-back turns: one person continuing is one turn
END_THRESHOLD = 3       # after MIN_TURNS, if every bid is below this, the meeting ends
LEAK_RETRIES = 2        # times a turn is regenerated if it gives the seed away
SPEECH_PASS = True       # after the meeting, rewrite each turn to sound spoken
SPEECH_PASS_MIN_WORDS = 15  # shorter turns are left as they are
SPEECH_PASS_LENGTH = (0.75, 1.3)  # rewrites outside this length ratio are discarded
REACTION_CHANCE = 0.3    # how often a listener's wish to react ("mm-hm") is let through

# Meeting clock, based on how much has actually been said,
# so time references stay consistent with the conversation.
MEETING_MINUTES = 10
WORDS_PER_MINUTE = 150

# Word budget per turn type. A turn more than LENGTH_SLACK over its
# budget is regenerated once, shorter.
WORD_LIMITS = {"respond": 80, "brief": 15, "interject": 40, "reaction": 8}
LENGTH_SLACK = 1.5

TURN_NOTES = {
    "respond": (
        "Respond to what was just said, or to an earlier point that "
        "matters to you. Usually one to three sentences; stay under "
        "about 80 words."
    ),
    "brief": (
        "Give a quick reaction: a few words, at most one short sentence "
        "(agreement, a short question, a hesitation)."
    ),
    "interject": (
        "You are jumping into the conversation. Jump straight to "
        "your point, the way people do when they interrupt. Keep it short."
    ),
    "reaction": (
        "You're not taking the floor, just reacting out loud as a listener "
        "while the conversation moves on: one to six words. It doesn't "
        "have to be agreement. It can be surprise (\"oh, really?\"), a "
        "quick check (\"wait, the 18th?\"), a small worry (\"hm, that's "
        "tight\"), a laugh, or just \"okay\". Don't evaluate or sum up "
        "what they said, don't add a point, and don't stamp approval "
        "(\"exactly\", \"agreed\", \"that works\")."
    ),
}
BID_INTENTS = ("respond", "brief", "interject")


def prepare_private_briefs(content, casting, qa_feedback=None):
    """Prepare a separate discussion brief for each participant."""

    planner_prompt = """
Prepare shared meeting background and one private brief per participant.
Use benchmark_task and seed_instruction to create the required situation.
Do not assume that it involves terminology or a hidden problem.
Give each speaker only their own knowledge, priorities, assumptions and concerns.
Never mention testing or give everyone the full reference answer.
For implicit conflicts, distribute facts without explaining the conflict.
For explicit tasks, permit definitions, objections, decisions and assignments.
For controls, do not plant the target problem or manufacture friction.
For unassigned action items, leave the intended ownership gaps intact.
Use concrete consistent details, and preserve the task's required facts.
Shared context contains only ordinary background everyone knows, not the answer.
Private briefs must not contain evaluator instructions or the full answer.
Do not prescribe dialogue or speaking order. Use QA feedback to improve briefs.
Return JSON:
{"shared_context": "ordinary shared background",
 "briefs": {"exact speaker name": "short private discussion brief"}}
"""

    setup = {
        "benchmark_category": content.seed.benchmark_category,
        "research_problem": content.research_problem,
        "meeting_context": content.meeting_context,
        "speakers": [
            speaker.model_dump(mode="json")
            for speaker in casting.speakers
        ],
        "benchmark_task": content.seed.benchmark_task,
        "seed_instruction": content.seed.explicit_instruction,
        "ground_truth": content.seed.ground_truth,
        "qa_feedback": qa_feedback,
    }

    result = call_llm_json(
        planner_prompt,
        json.dumps(setup, indent=2),
        max_tokens=3500,
    )

    shared_context = result.get("shared_context")
    if not isinstance(shared_context, str) or not shared_context.strip():
        raise ValueError("Planner returned no shared meeting context.")

    briefs = result.get("briefs", {})

    # Make sure every participant has a usable brief.
    for speaker in casting.speakers:
        brief = briefs.get(speaker.name)
        if not isinstance(brief, str) or not brief.strip():
            raise ValueError(
                f"Missing private brief for {speaker.name}"
            )

    return shared_context, briefs


def format_history(transcript):
    if not transcript:
        return "The meeting has not started yet."
    return "\n".join(f"{turn.speaker}: {turn.text}" for turn in transcript)


def time_cue(transcript):
    """
    Meeting time from words spoken so far, so agents wind down on their
    own and any time they mention matches the conversation. Near the turn
    cap, someone has a hard stop, which ends the meeting without
    contradicting the clock.
    """
    words = sum(len(turn.text.split()) for turn in transcript)
    minutes = round(words / WORDS_PER_MINUTE)
    cue = f"About {minutes} of {MEETING_MINUTES} scheduled minutes have passed."

    turns = len(transcript)
    if turns >= MAX_TURNS - 3 or minutes >= MEETING_MINUTES:
        cue += " Someone has a hard stop, so the meeting has to wrap up now."
    elif turns >= MAX_TURNS - 6 or minutes >= MEETING_MINUTES * 0.75:
        cue += " The meeting is getting close to its end."
    return cue



def participant_blocks(speaker, content, shared_context, brief, transcript, tail):
    """
    What one participant can see: their own profile and brief, plus the
    shared meeting. Built as content blocks so the parts that repeat
    across calls are cached:
      1. fixed context (profile, brief, ...) — identical on every call
      2. one block per turn so far — each call only adds the newest turn
      3. tail (time cue, style notes, instruction) — never cached
    """

    fixed = f"""
YOUR PROFILE:
{speaker.model_dump_json(indent=2)}

RESEARCH PROBLEM:
{content.research_problem}

MEETING CONTEXT:
{content.meeting_context}

SHARED BACKGROUND (everyone in the meeting knows this):
{shared_context}

YOUR PRIVATE DISCUSSION BRIEF:
{brief}

CONVERSATION SO FAR:"""

    blocks = [{"type": "text", "text": fixed, "prompt_cache_breakpoint": CACHE_BREAKPOINT}]

    if transcript:
        blocks += [
            {"type": "text", "text": f"{turn.speaker}: {turn.text}"}
            for turn in transcript
        ]
        blocks[-1]["prompt_cache_breakpoint"] = CACHE_BREAKPOINT
    else:
        blocks.append({"type": "text", "text": "The meeting has not started yet."})

    blocks.append({"type": "text", "text": tail})
    return blocks


def get_bid(speaker, content, shared_context, brief, transcript, nudge=""):
    """Ask one participant how strongly they want to speak next."""

    tail = f"""
TIME:
{time_cue(transcript)}
{nudge}

You are {speaker.name}. Decide whether you want to speak next.
Return JSON with "urgency", "intent" and "react".
"""
    blocks = participant_blocks(
        speaker, content, shared_context, brief, transcript, tail
    )

    result = call_llm_json(BID_PROMPT, blocks, max_tokens=150)

    try:
        urgency = max(0, min(10, int(result.get("urgency", 0))))
    except (TypeError, ValueError):
        urgency = 0

    intent = result.get("intent", "respond")
    if intent not in BID_INTENTS:
        intent = "pass" if urgency == 0 else "respond"

    return {
        "urgency": urgency,
        "intent": intent,
        "react": result.get("react") is True,
    }


def collect_bids(speakers, content, shared_context, briefs, transcript, nudges):
    """Every participant bids at the same time."""
    with ThreadPoolExecutor(max_workers=len(speakers)) as pool:
        futures = {
            speaker.name: pool.submit(
                get_bid, speaker, content, shared_context,
                briefs[speaker.name], transcript,
                nudges.get(speaker.name, ""),
            )
            for speaker in speakers
        }
        return {name: future.result() for name, future in futures.items()}


def consecutive_turns(transcript, name):
    count = 0
    for turn in reversed(transcript):
        if turn.speaker != name:
            break
        count += 1
    return count


def pick_speaker(bids, transcript, turn_index, can_end=True):
    """
    Highest bid wins, ties broken at random. Returns None when the
    meeting should end because nobody has anything left to say.
    can_end=False keeps the meeting going (e.g. the seed hasn't come up yet).
    """

    eligible = {
        name: bid for name, bid in bids.items()
        if consecutive_turns(transcript, name) < MAX_CONSECUTIVE
    }

    top = max(bid["urgency"] for bid in eligible.values())

    if can_end and turn_index >= MIN_TURNS and top < END_THRESHOLD:
        return None

    winners = [
        name for name, bid in eligible.items()
        if bid["urgency"] == top
    ]
    return random.choice(winners)


def pick_reactor(bids, transcript, winner, just_reacted):
    """
    A listener who wants to react to the last turn ("mm-hm", "wait, the
    18th?") gets a quick turn before the winner speaks, without needing
    to win the floor. Not the last speaker, and not the winner.
    With 3 speakers that leaves one person, who then became a
    backchannel machine, so nobody reacts twice without taking the
    floor in between (just_reacted).
    """
    if not transcript or random.random() > REACTION_CHANCE:
        return None
    reactors = [
        name for name, bid in bids.items()
        if bid["react"] and name not in (winner, transcript[-1].speaker)
        and name not in just_reacted
    ]
    return random.choice(reactors) if reactors else None


def check_seed_leak(content, transcript, text):
    """Check whether the turn violates the selected task's conditions."""

    task = {
        "benchmark_task": content.seed.benchmark_task,
        "seed_instruction": content.seed.explicit_instruction,
        "ground_truth": content.seed.ground_truth,
        "recent_conversation": format_history(transcript),
        "new_turn": text,
    }

    result = call_llm_json(
        LEAK_CHECK_PROMPT,
        json.dumps(task, indent=2),
        max_tokens=200,
    )
    if not isinstance(result.get("leaked"), bool):
        raise ValueError("Leak checker must return a boolean leaked field")
    return result["leaked"], result.get("reason", "")


def speak(speaker, content, shared_context, brief, ami_style_notes,
          transcript, intent, extra_note=""):
    """The winning participant writes their turn."""

    instruction = TURN_NOTES[intent]
    if not transcript:
        instruction = (
            "You're opening the meeting. Meetings rarely start on the "
            "agenda: there's usually a bit of small talk or logistics "
            "first. Then get things going around what you want out of "
            "today. Keep it short."
        )
    if extra_note:
        instruction += "\n" + extra_note

    def generate(note):
        tail = f"""
TIME:
{time_cue(transcript)}

STYLE REFERENCE (how people in a real meeting talked):
{ami_style_notes}

CURRENT INSTRUCTION:
{note}

Speak only as {speaker.name}.
Return JSON with the field "text".
"""
        blocks = participant_blocks(
            speaker, content, shared_context, brief, transcript, tail
        )
        result = call_llm_json(SYSTEM_PROMPT, blocks, max_tokens=700)

        text = result.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty or invalid response from {speaker.name}")
        return text.strip()

    text = generate(instruction)

    # Spoken turns are short. One retry if this reads like a written paragraph.
    limit = WORD_LIMITS[intent]
    if len(text.split()) > limit * LENGTH_SLACK:
        print(f"      ({speaker.name}: {len(text.split())} words, asking for a shorter turn)")
        text = generate(
            instruction
            + f"\nYour draft was {len(text.split())} words, far too long to "
            f"say out loud here. Say it in at most {limit} words, the way "
            "you'd actually say it."
        )

    return text


# One is picked at random for each rewritten turn. Asking for "a restart"
# in every long turn gave the same "X- well, X" repair everywhere.
DISFLUENCIES = [
    "none",
    "none",
    "abandon a sentence and start it again differently",
    "correct a word or number just said",
    "repeat a word while thinking",
    "lose the thread partway and come back to it",
    "trail off at the end instead of finishing the point",
    "search for a word",
]
# No example phrases here: the model copied them word for word.


def speech_pass(content, transcript):
    """
    The turns were written one at a time and still read like writing.
    Rewrite each one as spoken, keeping its content. A rewrite is
    discarded if its length changes too much or it violates task conditions.
    Final QA checks evidence after all rewrites.
    """

    def rewrite(i):
        turn = transcript[i]
        if len(turn.text.split()) < SPEECH_PASS_MIN_WORDS:
            return turn, None

        task = {
            "speaker": turn.speaker,
            "previous_turns": format_history(transcript[max(0, i - 3):i]),
            "benchmark_task": content.seed.benchmark_task,
            "disfluency": random.choice(DISFLUENCIES),
            "turn": turn.text,
        }
        result = call_llm_json(
            SPEECH_PASS_PROMPT, json.dumps(task, indent=2), max_tokens=400,
        )
        text = str(result.get("text", "")).strip()

        if not text:
            return turn, "empty"
        ratio = len(text.split()) / len(turn.text.split())
        if not SPEECH_PASS_LENGTH[0] <= ratio <= SPEECH_PASS_LENGTH[1]:
            return turn, "length changed"
        if turn.text.endswith("—") and not text.endswith("—"):
            text = text.rstrip(".!? ") + "—"
        leaked, _ = check_seed_leak(content, transcript[:i], text)
        if leaked:
            return turn, "task violation"
        return Turn(speaker=turn.speaker, text=text), None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(rewrite, range(len(transcript))))

    kept = [reason for _, reason in results if reason]
    rewritten = sum(
        1 for (new, reason), old in zip(results, transcript)
        if not reason and new.text != old.text
    )
    print(
        f"      Speech pass: rewrote {rewritten} of {len(transcript)} turns"
        + (f" (kept {len(kept)} originals: {', '.join(sorted(set(kept)))})"
           if kept else "")
    )
    return [new for new, _ in results]


def locate_seed(content, transcript):
    """Find evidence of the seed in the actual generated conversation."""

    locator_prompt = """
You inspect a generated meeting transcript for a benchmark feature.

Check the actual dialogue against the requested seed and ground truth.
Do not assume that the requested feature appeared.

Return JSON:
{
  "seed_location": "Relevant turn numbers and a short explanation"
}

If the task conditions are missing or incomplete, state that clearly.
Explicit statements are valid unless this task requires an unnoticed issue.
For controls, verify consistency and absence of the target problem.
Do not invent evidence.

"""

    numbered_turns = [
        {
            "turn": index + 1,
            "speaker": turn.speaker,
            "text": turn.text,
        }
        for index, turn in enumerate(transcript)
    ]

    task = {
        "benchmark_task": content.seed.benchmark_task,
        "seed_instruction": content.seed.explicit_instruction,
        "ground_truth": content.seed.ground_truth,
        "transcript": numbered_turns,
    }
    result = call_llm_json(
        locator_prompt, json.dumps(task, indent=2), max_tokens=1200,
    )

    location = result.get("seed_location")

    if not isinstance(location, str) or not location.strip():
        raise ValueError("Seed inspection returned no location report.")

    return location


def run(
    content: ContentCurationOutput,
    casting: CastingOutput,
    ami_style_notes: str,
    qa_feedback: str = None,
) -> FilmingOutput:
    """Simulate the meeting: participants bid for the floor each turn."""

    speakers = casting.speakers

    if len(speakers) < 2:
        raise ValueError("The simulation needs at least two speakers.")

    names = [speaker.name for speaker in speakers]
    if len(set(names)) != len(names):
        raise ValueError("Speaker names must be unique.")

    by_name = {speaker.name: speaker for speaker in speakers}

    shared_context, private_briefs = prepare_private_briefs(
        content,
        casting,
        qa_feedback,
    )

    transcript = []
    reactions = []          # texts of listener reactions so far
    just_reacted = set()    # reacted and hasn't taken the floor since

    # Task evidence is checked by QA; no word-count proxy for completion.
    for turn_index in range(MAX_TURNS):
        nudges = {}

        bids = collect_bids(
            speakers, content, shared_context, private_briefs, transcript,
            nudges,
        )
        bid_summary = ", ".join(
            f"{name.split()[-1]} {bid['urgency']}"
            for name, bid in bids.items()
        )

        name = pick_speaker(bids, transcript, turn_index, can_end=True)
        if name is None:
            print(f"      Meeting ended naturally (bids: {bid_summary})")
            break

        speaker = by_name[name]
        intent = bids[name]["intent"]
        if intent == "pass":
            intent = "respond"

        # Do not truncate previous facts to simulate an interruption.
        if intent != "interject" and len(transcript) < MAX_TURNS - 1:
            reactor = pick_reactor(bids, transcript, name, just_reacted)
            if reactor:
                # Reactions came out as "Mm-hm, yeah" over and over.
                recent = ", ".join(f'"{r}"' for r in reactions[-6:])
                reaction = speak(
                    by_name[reactor], content, shared_context,
                    private_briefs[reactor], ami_style_notes, transcript,
                    "reaction",
                    extra_note=(
                        f"Reactions already heard in this meeting: {recent}. "
                        "Don't reuse any of them or their wording."
                        if recent else ""
                    ),
                )
                if not check_seed_leak(content, transcript, reaction)[0]:
                    transcript.append(Turn(speaker=reactor, text=reaction))
                    reactions.append(reaction)
                    just_reacted.add(reactor)
                    print(f"      Turn {len(transcript)}: {reactor} (reaction)")

        nudge = nudges.get(name, "")
        text = speak(
            speaker, content, shared_context, private_briefs[name],
            ami_style_notes, transcript, intent, extra_note=nudge,
        )

        for attempt in range(LEAK_RETRIES + 1):
            leaked, reason = check_seed_leak(content, transcript, text)
            if not leaked:
                break
            if attempt == LEAK_RETRIES:
                print(f"      Turn {len(transcript) + 1}: seed still leaks, keeping it for QA ({reason})")
                break
            print(f"      Turn {len(transcript) + 1}: seed leak, regenerating ({reason})")
            text = speak(
                speaker, content, shared_context, private_briefs[name],
                ami_style_notes, transcript, intent,
                extra_note=(
                    "Stay within your own brief and perspective. Do not invent "
                    "facts or resolve matters you do not have evidence for. "
                    "Produce a different natural contribution.\n" + nudge
                ).strip(),
            )

        transcript.append(Turn(speaker=name, text=text))
        just_reacted.discard(name)

        print(
            f"      Turn {len(transcript)}: {name} ({intent}) "
            f"| bids: {bid_summary}"
        )

        if len(transcript) >= MAX_TURNS:
            break

    if SPEECH_PASS:
        transcript = speech_pass(content, transcript)

    seed_location = locate_seed(content, transcript)

    return FilmingOutput(
        transcript=transcript,
        seed_location=seed_location,
    )