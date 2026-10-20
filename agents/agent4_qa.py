"""
agent4_qa.py
QA Agent — LLM-as-judge that scores the transcript and
triggers one re-film cycle if it doesn't pass.
"""

import sys
sys.path.append("..")

from schema import QAOutput, QAFeedback, FilmingOutput, ContentCurationOutput, CastingOutput
from utils.llm import call_llm_json
from prompts.agent4_prompt import SYSTEM_PROMPT
import agents.agent3_filming as agent3


def run(
    filming: FilmingOutput,
    content: ContentCurationOutput,
    casting: CastingOutput,
    ami_style_notes: str,
) -> QAOutput:
    """
    Input:
        filming         — output of Agent 3 to evaluate
        content         — output of Agent 1 (for seed reference)
        casting         — output of Agent 2 (for speaker reference)
        ami_style_notes — for naturalness comparison

    Output:
        QAOutput with feedback + revised transcript if needed
    """

    # format transcript for judge
    transcript_text = "\n".join([
        f"{t.speaker}: {t.text}"
        for t in filming.transcript
    ])

    user_message = f"""
Evaluate this interdisciplinary research meeting transcript.

TRANSCRIPT:
{transcript_text}

SEED that should be embedded:
{content.seed.explicit_instruction}

SEED location reported by Filming agent:
{filming.seed_location}

AMI style reference (what a natural meeting sounds like):
{ami_style_notes}

Score each dimension 1-5 and list specific issues.
Set approved to true only if all scores >= 3 AND seed_embedded is true.

Return JSON:
{{
  "topical_alignment": 1-5,
  "conversational_naturalness": 1-5,
  "dialogue_flow": 1-5,
  "seed_embedded": true|false,
  "seed_naturalness": 1-5,
  "issues": ["issue 1", "issue 2"],
  "approved": true|false
}}
"""

    raw = call_llm_json(SYSTEM_PROMPT, user_message)

    feedback = QAFeedback(
        topical_alignment=raw.get("topical_alignment", 3),
        conversational_naturalness=raw.get("conversational_naturalness", 3),
        dialogue_flow=raw.get("dialogue_flow", 3),
        seed_embedded=raw.get("seed_embedded", False),
        seed_naturalness=raw.get("seed_naturalness", 3),
        issues=raw.get("issues", []),
        approved=raw.get("approved", False),
    )

    # one re-film cycle if not approved
    revised = None
    if not feedback.approved:
        print("[qa] Transcript did not pass. Triggering re-film...")
        issues_text = "\n".join(f"- {i}" for i in feedback.issues)
        revised = agent3.run(
            content=content,
            casting=casting,
            ami_style_notes=ami_style_notes,
            qa_feedback=issues_text,
        )

    return QAOutput(feedback=feedback, revised_transcript=revised)
