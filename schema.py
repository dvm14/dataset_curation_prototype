"""
schema.py
Pydantic models for the benchmark transcript generation pipeline.
All agents use these types as their input/output contract.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum


# ─────────────────────────────────────────────
# ENUMS — value banks for speaker attributes
# ─────────────────────────────────────────────

class FunctionalRole(str, Enum):
    PI               = "principal investigator"
    POSTDOC          = "postdoctoral researcher"
    PHD_STUDENT      = "PhD student"
    RESEARCH_ENGINEER = "research engineer"
    DOMAIN_EXPERT    = "domain expert"
    PROJECT_MANAGER  = "project manager"

class Tone(str, Enum):
    FORMAL      = "formal"
    INFORMAL    = "informal"
    CAUTIOUS    = "cautious"
    ENTHUSIASTIC = "enthusiastic"
    DIPLOMATIC  = "diplomatic"

class LanguageComplexity(str, Enum):
    HIGH_JARGON   = "high jargon — uses discipline-specific terminology freely"
    MODERATE      = "moderate — balances jargon with accessible language"
    PLAIN         = "plain — avoids jargon, explains concepts simply"

class CommunicationStyle(str, Enum):
    ASSERTIVE   = "assertive"
    COLLABORATIVE = "collaborative"
    QUESTIONING = "questioning"
    RESERVED    = "reserved"
    DOMINANT    = "dominant"

class EpistemicStance(str, Enum):
    CONFIDENT   = "highly confident in their discipline's contribution"
    OPEN        = "open to other perspectives, willing to revise"
    UNCERTAIN   = "uncertain about how their methods apply to this problem"
    SKEPTICAL   = "skeptical of other disciplines' approaches"
    DEFERENTIAL = "defers to other experts outside their domain"

class DisciplineDistance(str, Enum):
    CLOSE    = "close"    # e.g. biology + biochemistry
    DISTANT  = "distant"  # e.g. ecology + economics
    MIXED    = "mixed"    # some close, some distant pairs


# ─────────────────────────────────────────────
# STAGE 1 — AMI STYLE INPUT
# ─────────────────────────────────────────────

class AMIStyleTemplate(BaseModel):
    """
    A real AMI meeting excerpt used as a style reference.
    Content is not reused — only tone, turn-taking, and
    conversational dynamics are extracted.
    """
    raw_transcript: str = Field(
        description="Raw text of the AMI meeting excerpt"
    )
    extracted_style_notes: Optional[str] = Field(
        default=None,
        description="LLM-extracted style observations: turn length, "
                    "interruption frequency, hedging language, formality level"
    )


# ─────────────────────────────────────────────
# AGENT 1 — CONTENT CURATION OUTPUT
# ─────────────────────────────────────────────

class SeedSpec(BaseModel):
    """
    The explicit seed instruction generated from the benchmark task.
    Tells the Filming agent exactly what interdisciplinary feature
    to embed and where.
    """
    benchmark_category: str = Field(
        description="Which benchmark category this seed tests "
                    "(e.g. jargon, assumptions, disagreement)"
    )
    benchmark_task: str = Field(
        description="The original benchmark task description"
    )
    explicit_instruction: str = Field(
        description="The concrete injection instruction for the Filming agent. "
                    "e.g. 'At some point in the discussion, the ecologist and "
                    "the economist both use the word resilience but mean "
                    "different things...'"
    )
    ground_truth: dict = Field(
        description="The known correct answer for evaluation. "
                    "e.g. {'term': 'resilience', "
                    "'discipline_a': 'ecology', "
                    "'definition_a': 'ecological recovery after disturbance', "
                    "'discipline_b': 'economics', "
                    "'definition_b': 'ability of a market to absorb shocks'}"
    )

class ContentCurationOutput(BaseModel):
    """Output of Agent 1 — Content Curation"""
    research_problem: str = Field(
        description="The interdisciplinary research problem the meeting addresses"
    )
    disciplines: list[str] = Field(
        description="List of disciplines present in the meeting"
    )
    discipline_distance: DisciplineDistance
    meeting_context: str = Field(
        description="Brief background on the team, project stage, "
                    "and what this meeting is trying to accomplish"
    )
    seed: SeedSpec = Field(
        description="The embedded benchmark feature for this transcript"
    )


# ─────────────────────────────────────────────
# AGENT 2 — CASTING OUTPUT
# ─────────────────────────────────────────────

class SpeakerProfile(BaseModel):
    """Profile for a single meeting participant"""
    name: str
    discipline: str
    functional_role: FunctionalRole
    background: str = Field(
        description="1-2 sentences on experience and qualifications"
    )
    distinct_perspective: str = Field(
        description="How this speaker frames the research problem "
                    "through their disciplinary lens"
    )
    tone: Tone
    language_complexity: LanguageComplexity
    communication_style: CommunicationStyle
    epistemic_stance: EpistemicStance

class CastingOutput(BaseModel):
    """Output of Agent 2 — Casting"""
    speakers: list[SpeakerProfile] = Field(
        description="All meeting participants, one per discipline "
                    "unless specified otherwise",
        min_length=2
    )


# ─────────────────────────────────────────────
# AGENT 3 — FILMING OUTPUT
# ─────────────────────────────────────────────

class Turn(BaseModel):
    """A single speaking turn in the transcript"""
    speaker: str = Field(description="Speaker name")
    text: str    = Field(description="What they said")

class FilmingOutput(BaseModel):
    """Output of Agent 3 — Filming"""
    transcript: list[Turn] = Field(
        description="Ordered list of speaking turns, ~20-30 turns "
                    "for a 10-minute meeting"
    )
    seed_location: str = Field(
        description="Brief note on where in the transcript the seed "
                    "was embedded, for ground truth logging"
    )


# ─────────────────────────────────────────────
# AGENT 4 — QA OUTPUT
# ─────────────────────────────────────────────

class QAFeedback(BaseModel):
    """Structured feedback from the QA agent / LLM judge"""
    topical_alignment: int = Field(
        ge=1, le=5,
        description="Does the transcript stay on the research topic? (1-5)"
    )
    conversational_naturalness: int = Field(
        ge=1, le=5,
        description="Does it sound like a real meeting? Turn-taking, "
                    "interruptions, hedging? (1-5)"
    )
    dialogue_flow: int = Field(
        ge=1, le=5,
        description="Does the conversation progress coherently? (1-5)"
    )
    seed_embedded: bool = Field(
        description="Is the benchmark seed clearly present in the transcript?"
    )
    seed_naturalness: int = Field(
        ge=1, le=5,
        description="Does the seed feel natural or forced? (1-5)"
    )
    issues: list[str] = Field(
        description="List of specific problems to fix in re-filming"
    )
    approved: bool = Field(
        description="True if transcript passes QA, False if re-filming needed"
    )

class QAOutput(BaseModel):
    """Output of Agent 4 — QA"""
    feedback: QAFeedback
    revised_transcript: Optional[FilmingOutput] = Field(
        default=None,
        description="Re-filmed transcript if approved=False, else None"
    )


# ─────────────────────────────────────────────
# FINAL OUTPUT — saved to outputs/
# ─────────────────────────────────────────────

class GeneratedTranscript(BaseModel):
    """
    The complete pipeline output saved to outputs/.
    Includes the transcript + all metadata needed for evaluation.
    """
    transcript_id: str
    generated_at: str
    
    # pipeline inputs
    ami_style_notes: str
    research_problem: str
    disciplines: list[str]
    discipline_distance: DisciplineDistance
    speakers: list[SpeakerProfile]
    
    # benchmark metadata
    benchmark_category: str
    benchmark_task: str
    seed_spec: SeedSpec
    seed_location: str
    ground_truth: dict
    
    # the transcript itself
    transcript: list[Turn]
    
    # qa record
    qa_feedback: QAFeedback
    qa_cycles: int = Field(description="How many QA/re-film cycles were needed")
