import json

from schema import (
    FilmingOutput,
    Turn,
    ContentCurationOutput,
    CastingOutput,
)
from utils.llm import call_llm_json
from prompts.agent3_prompt import SYSTEM_PROMPT


# One turn means one participant speaks once.
MAX_TURNS = 24


def prepare_private_briefs(content, casting, qa_feedback=None):
    """Prepare a separate discussion brief for each participant."""

    planner_prompt = """
You prepare private briefs for participants in a research meeting.

Use the meeting scenario, speaker profiles, and benchmark seed to
give each participant relevant knowledge, priorities, or assumptions.

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

Return JSON:
{
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
        max_tokens=2500,
    )

    briefs = result.get("briefs", {})

    # Make sure every participant has a usable brief.
    for speaker in casting.speakers:
        brief = briefs.get(speaker.name)
        if not isinstance(brief, str) or not brief.strip():
            raise ValueError(
                f"Missing private brief for {speaker.name}"
            )

    return briefs


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
    """Generate a meeting through separate participant turns."""

    speakers = casting.speakers

    if len(speakers) < 2:
        raise ValueError("The simulation needs at least two speakers.")

    names = [speaker.name for speaker in speakers]
    if len(set(names)) != len(names):
        raise ValueError("Speaker names must be unique.")

    private_briefs = prepare_private_briefs(
        content,
        casting,
        qa_feedback,
    )

    transcript = []

    # Ensure everyone can speak at least once.
    total_turns = max(MAX_TURNS, len(speakers))

    for turn_index in range(total_turns):
        # First version: participants take turns in a fixed order.
        speaker = speakers[turn_index % len(speakers)]

        history = "\n".join(
            f"{turn.speaker}: {turn.text}"
            for turn in transcript
        )

        if not history:
            history = "The meeting has not started yet."

        if turn_index == 0:
            stage_note = (
                "Open the discussion with a concrete question or "
                "proposal related to the meeting goal."
            )
        elif turn_index >= total_turns - len(speakers):
            stage_note = (
                "The meeting is nearing its end. Respond to the "
                "discussion and help clarify next steps or remaining "
                "concerns. Do not invent consensus."
            )
        else:
            stage_note = (
                "Continue the discussion. Respond to a relevant "
                "point and contribute something useful."
            )

        user_message = f"""
YOUR PROFILE:
{speaker.model_dump_json(indent=2)}

RESEARCH PROBLEM:
{content.research_problem}

MEETING CONTEXT:
{content.meeting_context}

YOUR PRIVATE DISCUSSION BRIEF:
{private_briefs[speaker.name]}

STYLE REFERENCE:
{ami_style_notes}

CONVERSATION SO FAR:
{history}

CURRENT INSTRUCTION:
{stage_note}

Speak only as {speaker.name}.
Return JSON with the field "text".
"""

        result = call_llm_json(
            SYSTEM_PROMPT,
            user_message,
            max_tokens=700,
        )

        text = result.get("text")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Empty or invalid response from {speaker.name}"
            )

        transcript.append(
            Turn(
                speaker=speaker.name,
                text=text.strip(),
            )
        )

        print(
            f"      Turn {turn_index + 1}/{total_turns}: "
            f"{speaker.name}"
        )

    seed_location = locate_seed(content, transcript)

    return FilmingOutput(
        transcript=transcript,
        seed_location=seed_location,
    )