"""
agent3_filming.py
Filming Agent — simulates the meeting. Each cast speaker is a
separate agent with a private brief. Nobody assigns turns: every
round, each agent bids for the floor and the strongest bid speaks.
"""

import json
import random
import re
from concurrent.futures import ThreadPoolExecutor

from schema import (
    FilmingOutput,
    Turn,
    ContentCurationOutput,
    CastingOutput,
)
from utils.llm import call_llm_json
from prompts.agent3_prompt import SYSTEM_PROMPT, BID_PROMPT, LEAK_CHECK_PROMPT


# One turn means one participant speaks once.
MIN_TURNS = 15          # the meeting can't end on its own before this
MAX_TURNS = 35          # hard stop
MAX_CONSECUTIVE = 1     # no back-to-back turns: one person continuing is one turn
END_THRESHOLD = 3       # after MIN_TURNS, if every bid is below this, the meeting ends
LEAK_RETRIES = 2        # times a turn is regenerated if it gives the seed away
NUDGE_AFTER = 8         # from this turn on, seed speakers who haven't said the term get a private nudge

# Meeting clock, based on how much has actually been said,
# so time references stay consistent with the conversation.
MEETING_MINUTES = 10
WORDS_PER_MINUTE = 150

# Word budget per turn type. A turn more than LENGTH_SLACK over its
# budget is regenerated once, shorter.
WORD_LIMITS = {"respond": 80, "brief": 15, "interject": 40}
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
        "You just cut the last speaker off mid-sentence. Jump straight to "
        "your point, the way people do when they interrupt. Keep it short."
    ),
}


def prepare_private_briefs(content, casting, qa_feedback=None):
    """Prepare a separate discussion brief for each participant."""

    planner_prompt = """
You prepare a research meeting: shared background that everyone in
the room knows, plus a private brief for each participant.

Use the meeting scenario, speaker profiles, and benchmark seed to
give each participant relevant knowledge, priorities, or assumptions.

Shared background (everyone sees the same text):
- Ground the meeting in concrete, consistent specifics: the city and
  neighborhoods, the institution(s), the grant or project name and its
  deadline, today's date, datasets by name, and what happened at the
  last meeting or since.
- Include a logistics detail or two (someone joined late, a deadline
  moved, a room or call issue) so the meeting isn't purely on-agenda.
- Don't mention the seed here.

Private briefs:
- Disagreements should only come from genuine differences in the
  speakers' priorities. Don't add conflict for its own sake.

Rules:
- Give each speaker only their own perspective.
- Do not reveal the full benchmark answer to every participant.
- Do not tell participants that they are being tested.
- Do not write dialogue or prescribe an exact speaking sequence.
- For an ambiguous term, give each relevant speaker their own
  discipline's meaning and a reason to use that term naturally.
- Do not tell speakers that another discipline uses a different meaning.
- Keep each brief short and relevant to the meeting.
- Treat QA feedback as guidance for preparing better briefs.

Keeping the seed invisible:
- Write each speaker's meaning as simply how they think about the
  problem, the way it would appear in their own notes. It should feel
  obvious to them, not like one option among several.
- Never use words like "ambiguous", "different meaning", "definition",
  "terminology", or "be careful how you use".
- Don't put the seed term in quotes or single it out for emphasis.
  Work it into the speaker's goals and concerns so they say it naturally.
- Each speaker involved in the seed must have the exact seed term in
  their brief several times, used in their own sense, so it is clearly
  their everyday word for that idea (not a synonym like "graph" for
  "network").
- Speakers not involved in the seed get an ordinary brief about their
  own priorities, with nothing that would make them examine the term.

Return JSON:
{
  "shared_context": "background everyone in the meeting knows",
  "briefs": {
    "exact speaker name": "private brief for this speaker"
  }
}
"""

    setup = {
        "research_problem": content.research_problem,
        "meeting_context": content.meeting_context,
        "speakers": [
            speaker.model_dump(mode="json")
            for speaker in casting.speakers
        ],
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


CACHE = {"type": "ephemeral"}


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

    blocks = [{"type": "text", "text": fixed, "cache_control": CACHE}]

    if transcript:
        blocks += [
            {"type": "text", "text": f"{turn.speaker}: {turn.text}"}
            for turn in transcript
        ]
        blocks[-1]["cache_control"] = CACHE
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
Return JSON with "urgency" and "intent".
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
    if intent not in TURN_NOTES:
        intent = "pass" if urgency == 0 else "respond"

    return {"urgency": urgency, "intent": intent}


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


def seed_terms(ground_truth):
    """
    The seed's key term(s), when the ground truth names one. Agent 1 isn't
    consistent about the key ("term", "ambiguous_term", ...), so take any
    short string value under a key containing "term".
    """
    terms = []

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if ("term" in key.lower() and isinstance(item, str)
                        and 0 < len(item.strip()) <= 40):
                    terms.append(item.strip())
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(ground_truth)
    return list(dict.fromkeys(terms))


def mentions(text, terms):
    """Whole-word match that also allows a plural ("networks")."""
    return any(
        re.search(rf"\b{re.escape(term)}(s|es)?\b", text, re.IGNORECASE)
        for term in terms
    )


def seed_speakers(content, speakers):
    """Speakers the seed involves: named (full or last name) in the seed spec."""
    spec = content.seed.explicit_instruction + json.dumps(content.seed.ground_truth)
    return [
        speaker for speaker in speakers
        if speaker.name in spec or speaker.name.split()[-1] in spec
    ]


def seed_nudge(terms):
    """Private reminder for a seed speaker who hasn't used the term yet."""
    return (
        "Something from your brief you haven't raised yet involves what you "
        f"call {terms[0]}. If it fits the conversation, bring it up now, "
        f"using the word \"{terms[0]}\" the way you normally would."
    )


def check_seed_leak(content, transcript, text):
    """Does this new turn point out the hidden feature instead of just containing it?"""

    terms = seed_terms(content.seed.ground_truth)
    if terms and not mentions(text, terms):
        return False, ""

    task = {
        "seed_instruction": content.seed.explicit_instruction,
        "ground_truth": content.seed.ground_truth,
        "recent_conversation": format_history(transcript[-6:]),
        "new_turn": text,
    }

    result = call_llm_json(
        LEAK_CHECK_PROMPT,
        json.dumps(task, indent=2),
        max_tokens=200,
    )
    return bool(result.get("leaked", False)), result.get("reason", "")


def cut_off(turn, protected_terms):
    """
    The interrupted speaker doesn't get to finish: cut their turn
    partway, at a comma if there is one. Turns that are too short to cut,
    or where the cut would remove the seed term, are left alone.
    """
    words = turn.text.split()
    if len(words) < 12:
        return turn

    low, high = int(len(words) * 0.4), int(len(words) * 0.75)
    at_comma = [i for i in range(low, high) if words[i - 1].endswith(",")]
    cut = random.choice(at_comma or list(range(low, high)))

    removed = " ".join(words[cut:]).lower()
    if any(term.lower() in removed for term in protected_terms):
        return turn

    kept = " ".join(words[:cut]).rstrip(",;:.")
    return Turn(speaker=turn.speaker, text=kept + "—")


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

If the feature is missing, incomplete, or explicitly revealed by
participants, state that clearly. Do not invent evidence.
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
        "seed_instruction": content.seed.explicit_instruction,
        "ground_truth": content.seed.ground_truth,
        "transcript": numbered_turns,
    }

    result = call_llm_json(
        locator_prompt,
        json.dumps(task, indent=2),
        max_tokens=1200,
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

    # Make sure the seed actually comes up: track which seed speakers
    # have said the term, and privately nudge the ones who haven't.
    terms = seed_terms(content.seed.ground_truth)
    involved = seed_speakers(content, speakers) if terms else []

    def still_missing():
        return [
            s.name for s in involved
            if not any(t.speaker == s.name and mentions(t.text, terms)
                       for t in transcript)
        ]

    for turn_index in range(MAX_TURNS):
        missing = still_missing()
        nudges = {
            name: seed_nudge(terms)
            for name in missing if turn_index >= NUDGE_AFTER
        }

        bids = collect_bids(
            speakers, content, shared_context, private_briefs, transcript,
            nudges,
        )
        bid_summary = ", ".join(
            f"{name.split()[-1]} {bid['urgency']}"
            for name, bid in bids.items()
        )

        name = pick_speaker(bids, transcript, turn_index, can_end=not missing)
        if name is None:
            print(f"      Meeting ended naturally (bids: {bid_summary})")
            break

        speaker = by_name[name]
        intent = bids[name]["intent"]
        if intent == "pass":
            intent = "respond"

        if intent == "interject" and transcript:
            transcript[-1] = cut_off(transcript[-1], terms)

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
                print(f"      Turn {turn_index + 1}: seed still leaks, keeping it for QA ({reason})")
                break
            print(f"      Turn {turn_index + 1}: seed leak, regenerating ({reason})")
            text = speak(
                speaker, content, shared_context, private_briefs[name],
                ami_style_notes, transcript, intent,
                extra_note=(
                    "Stay on the substance of the project. Don't step back "
                    "to comment on how anyone is using particular words, "
                    "and don't ask people to define their terms.\n" + nudge
                ).strip(),
            )

        transcript.append(Turn(speaker=name, text=text))

        print(
            f"      Turn {turn_index + 1}: {name} ({intent}) "
            f"| bids: {bid_summary}"
        )

    if involved:
        missing = still_missing()
        print(
            f"      Seed term {terms}: "
            + ("said by every seed speaker" if not missing
               else f"never said by {', '.join(missing)}")
        )

    seed_location = locate_seed(content, transcript)

    return FilmingOutput(
        transcript=transcript,
        seed_location=seed_location,
    )
