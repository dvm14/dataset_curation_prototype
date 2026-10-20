"""
style_extractor.py
Extracts conversational style observations from a real AMI transcript.
Output is passed to the Filming agent as a style reference.
"""

import sys
sys.path.append("..")

from utils.llm import call_llm

SYSTEM_PROMPT = """
You are a linguist analyzing the conversational style of a real meeting transcript.
Extract style observations that a transcript writer could use to mimic this meeting's
natural feel — without reproducing any of the actual content.

Focus on:
- Average turn length (short/medium/long)
- Interruption and overlap frequency
- Hedging and uncertainty language patterns
- Formality level
- How participants build on or challenge each other
- Any notable speech patterns (filler words, false starts, etc.)
- How disagreement is expressed (direct vs. indirect)
- How questions are asked

Return a concise paragraph of style notes, not a list.
Do not reference the actual content or topic of the meeting.
"""

def extract_style(ami_transcript: str) -> str:
    """
    Takes a raw AMI transcript string.
    Returns a paragraph of style observations.
    """
    user_message = f"""
Analyze the conversational style of this meeting transcript:

{ami_transcript}

Return style observations only — no content summary.
"""
    return call_llm(SYSTEM_PROMPT, user_message)
