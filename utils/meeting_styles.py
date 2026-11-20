"""
meeting_styles.py
Bank of meeting styles and openings for the Filming agent.

Before this, every transcript was given the same style summary from one
AMI excerpt, whose first lines are people checking that they can log in.
That is why "can you hear me okay?" opened a third of the meetings, and why
most meetings had the same polished, agreeable shape. Each run now gets one
style and one opening picked at random, on top of the AMI style notes.

The example lines are invented and are not tied to any research topic.
Task instructions always win over the style (for example, a task that needs
one discipline to dominate overrides a "balanced" style).
"""

import random

MEETING_STYLES = {
    "quick_checkin": {
        "description": (
            "A rushed weekly check-in. People give short status updates and "
            "ask each other quick questions. Most turns are one or two "
            "sentences. Nobody summarizes. Someone has another meeting "
            "soon and the conversation stops where it stops."
        ),
        "examples": [
            "Okay, so, quick one from me- the export finished last night, mostly.",
            "Mostly?",
            "Two of the sites didn't come through. I haven't looked yet.",
        ],
    },
    "one_voice_leads": {
        "shows_seniority": True,
        "description": (
            "One person (often the most senior) talks for much of the "
            "meeting in long turns. Others come in with short questions or "
            "one-line agreement. A quieter person speaks rarely and "
            "usually only when asked directly or after a pause."
        ),
        "examples": [
            "So what I keep coming back to, and this is just from the last "
            "couple of projects I've run, is that you pay for it later if "
            "you skip the pilot, you always do, and then...",
            "Mm.",
            "Sorry, can I ask- is that the same pilot as the spring one?",
        ],
    },
    "thinking_aloud": {
        "description": (
            "An exploratory brainstorm. People ask questions they don't "
            "know the answer to, try out half-formed ideas, and sometimes "
            "contradict themselves. Nothing is decided. Turns are short to "
            "medium and often end in a question."
        ),
        "examples": [
            "I wonder if it's even the same thing- like, are we measuring "
            "the same thing in both places?",
            "Honestly no idea. Maybe? I'd have to look at how they set it up.",
            "Huh. Okay, so what if we just asked them?",
        ],
    },
    "tangent_heavy": {
        "description": (
            "The conversation drifts. Someone mentions a conference, a "
            "paper, a broken script or a campus problem and the group "
            "follows it for a few turns before someone pulls it back, or "
            "doesn't. The main topic still gets covered but unevenly."
        ),
        "examples": [
            "Wait, did that reviewer thing ever get sorted out?",
            "Oh god, no. Anyway, sorry- what were we saying about the sites?",
            "No, go on, I want to hear about the reviewer.",
        ],
    },
    "polite_friction": {
        "description": (
            "Courteous but with real tension. People disagree in an "
            "indirect way: with a question, a hedge, a long pause, or "
            "'let's take that offline'. Some disagreements are dropped "
            "rather than resolved. Nobody raises their voice."
        ),
        "examples": [
            "I mean, that could work. I'd just want to see it on our data first.",
            "Sure. We can park that and come back to it.",
            "Right. I'd still like to know how you'd handle the late ones, but okay.",
        ],
    },
    "round_robin_updates": {
        "description": (
            "Each person takes a turn reporting what they have done since "
            "last time. Updates are fairly long and mostly uninterrupted. "
            "There are a few clarifying questions between updates and "
            "little cross-talk. The discussion that does happen is brief."
        ),
        "examples": [
            "Okay, mine. I got through most of the interviews, I think "
            "fourteen of twenty, and the rest are scheduled for the "
            "following two weeks.",
            "Fourteen, nice. Were those all from the first batch?",
            "Most of them. A couple were referrals.",
        ],
    },
    "screen_share_working_session": {
        "description": (
            "Someone is sharing a document, figure or spreadsheet and the "
            "group works through it. People point at what is on screen "
            "('that column', 'the second figure'), but whoever is reading "
            "still says the key numbers, names and dates out loud, since "
            "only the audio is recorded. Turns are short and practical, "
            "with pauses while someone scrolls or reads."
        ),
        "examples": [
            "Can you scroll up- no, further. Yeah, that row.",
            "That one's wrong, I think. Or it's from the old file.",
            "Hang on, let me find it. Okay, give me a sec.",
        ],
    },
    "deferential_junior": {
        "shows_seniority": True,
        "description": (
            "A senior person leads and a more junior person is cautious, "
            "checks before offering ideas ('this might be a dumb question') "
            "and sometimes holds back something useful. A third person "
            "occasionally draws them out. The senior person is friendly "
            "but sometimes talks over them."
        ),
        "examples": [
            "This might be a dumb question, but is that normal for this kind of data?",
            "No, it's fine, it's- well, it depends. Go ahead.",
            "Sorry, I was going to say, in my group we usually-",
        ],
    },
    "running_out_of_time": {
        "description": (
            "The meeting starts late or the agenda is too long. People "
            "keep an eye on the clock, cut themselves off, skip items, and "
            "the meeting ends with things unfinished. No final recap."
        ),
        "examples": [
            "We've got maybe fifteen minutes, so- can we skip the intro stuff?",
            "Yeah, sure. Let's just start with what's blocking.",
            "I really have to jump in about five, sorry.",
        ],
    },
}

# How the meeting begins. The stock "can you hear me?" opener has been
# the most common start, so it is only one option among many and is
# explicitly discouraged unless chosen.
MEETING_OPENINGS = [
    "Start mid-thought, as if the conversation was already under way "
    "before the recording started.",
    "Start with one person apologizing for being a couple of minutes late "
    "and asking where the group has got to.",
    "Start with a quick question about something from last week, for example "
    "whether a file, form or email ever came through.",
    "Start with a few lines of small talk (weather, campus, a deadline "
    "elsewhere), then move to the topic without announcing it.",
    "Start with someone saying what they most want out of this meeting, "
    "briefly and casually.",
    "Start with one person already reading from, or referring to, a "
    "document the others have open.",
    "Start with a person asking for a quick catch-up because they missed "
    "the last meeting.",
]

STOCK_PHRASES_TO_AVOID = [
    "can you hear me okay",
    "lost my thread",
    "lost my train of thought",
    "lost my point",
    "can everyone hear me",
]


# Styles whose shape contradicts what a task needs. Found in the Claude pilot
# (e.g. assumptions_03 needs a long meeting and drew quick_checkin). Only
# clear contradictions are listed; anything else is left to the rule that
# task instructions win.
SHORT = ["quick_checkin", "running_out_of_time"]      # cut short, items skipped
TASK_STYLE_EXCLUDE = {
    "assumptions_03": SHORT,                            # "a long meeting"
    "problem_03": SHORT,                                # weigh questions, choose one
    "problem_04": SHORT + ["one_voice_leads"],          # each of three opens a question
    "coordination_01": ["running_out_of_time"],         # plan next steps
    "coordination_05": SHORT,                           # end with exactly six items
    "methods_02": SHORT + ["round_robin_updates"],      # credible reasons on both sides
    "disagreement_01": SHORT + ["round_robin_updates"], # rationale and evidence per side
    "other_01": ["polite_friction"],                    # control: productive, no tension
}


def pick_style(name=None, avoid_seniority=False, task_id=None):
    """
    Return (name, style). With name=None the style is chosen at random.

    avoid_seniority — True for tasks where the dialogue must not reveal who
    is senior (other_02, seniority_pair). Styles marked "shows_seniority"
    are never picked then, and asking for one by name is an error.
    task_id — styles in TASK_STYLE_EXCLUDE for that task are never picked.
    """
    if name is not None:
        if name not in MEETING_STYLES:
            raise ValueError(
                f"Unknown meeting_style {name!r}. Options: {sorted(MEETING_STYLES)}"
            )
        if avoid_seniority and MEETING_STYLES[name].get("shows_seniority"):
            raise ValueError(
                f"meeting_style {name!r} builds seniority into the dialogue, "
                "which this task forbids. Pick another style or leave it unset."
            )
        if name in TASK_STYLE_EXCLUDE.get(task_id, []):
            raise ValueError(f"meeting_style {name!r} contradicts task {task_id}.")
        return name, MEETING_STYLES[name]
    excluded = TASK_STYLE_EXCLUDE.get(task_id, [])
    options = [n for n, st in MEETING_STYLES.items()
               if not (avoid_seniority and st.get("shows_seniority"))
               and n not in excluded]
    name = random.choice(options)
    return name, MEETING_STYLES[name]


def pick_opening():
    """One opening instruction, chosen at random."""
    return random.choice(MEETING_OPENINGS)


def style_block(name, style, opening=None):
    """
    Text appended to the AMI style notes for one meeting.
    The Filming and QA agents already receive ami_style_notes, so this
    needs no change to either agent.
    """
    opening = opening or pick_opening()
    examples = "\n".join(f"  - {line}" for line in style["examples"])
    avoid = "; ".join(f'"{p}"' for p in STOCK_PHRASES_TO_AVOID)
    return (
        f"Meeting style for this meeting ({name}): {style['description']}\n"
        f"Illustration of the feel only (invented, do not reuse the wording or topic):\n"
        f"{examples}\n"
        f"Opening: {opening}\n"
        f"Do not use these stock phrases: {avoid}. The recording does not "
        f"need to begin with audio or login checks.\n"
        f"If the task instructions need something this style does not give "
        f"(for example one discipline dominating, or an item left open), "
        f"follow the task instructions."
    )
