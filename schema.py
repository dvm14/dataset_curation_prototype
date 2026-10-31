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
    # team rubric (1 poor / 3 mixed / 5 strong)
    seed_recoverable: int = Field(
        ge=1, le=5,
        description="Can evidence lines be pointed to, and does the ground "
                    "truth follow from them with no outside assumptions? (1-5)"
    )
    seed_implicit: int = Field(
        ge=1, le=5,
        description="Is the seeded issue left unnamed, unasked-about, and "
                    "not silently corrected, so the assistant has to find it? (1-5)"
    )
    naturalness: int = Field(
        ge=1, le=5,
        description="How many realism signals are present, and could a blind "
                    "reader tell it from a real meeting? (1-5)"
    )
    speaker_distinctness: int = Field(
        ge=1, le=5,
        description="Are speakers matchable to personas, with discipline-typical "
                    "vocabulary and no caricatures? (1-5)"
    )
    # pipeline checks not covered by the rubric
    domain_alignment: int = Field(
        ge=1, le=5,
        description="Does each speaker's contribution stay grounded in their "
                    "own discipline? The research question itself may shift. (1-5)"
    )
    dialogue_flow: int = Field(
        ge=1, le=5,
        description="Does the conversation progress coherently? (1-5)"
    )
    # evidence behind the scores
    evidence_turns: list[int] = Field(
        default_factory=list,
        description="Turn numbers (1-indexed) the ground truth can be derived from"
    )
    realism_signals: list[str] = Field(
        default_factory=list,
        description="Which realism signals the judge found in the transcript"
    )
    issues: list[str] = Field(
        description="List of specific problems to fix in re-filming"
    )
    seed_uses: list[dict] = Field(
        default_factory=list,
        description="Label for each turn with the seed term (and the turn after): "
                    "sense, comments_on_term, renames"
    )
    # finer naturalness measure, for comparing runs; not part of approval
    spoken_share: float | None = Field(
        default=None,
        description="Share of substantive turns the judge labels as sounding "
                    "spoken rather than written (0-1)"
    )
    written_turns: list[int] = Field(
        default_factory=list,
        description="Substantive turns labeled as reading like written prose"
    )
    approved: bool = Field(
        description="True if transcript passes QA, False if re-filming needed"
    )

class QAAttempt(BaseModel):
    """One film + QA round, kept so score changes across re-films are visible"""
    attempt: int = Field(description="1 = first filming, 2 = re-film")
    num_turns: int
    scores: dict[str, int] = Field(description="Rubric scores for this attempt")
    spoken_share: float | None = None
    approved: bool
    issues: list[str]


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
    qa_feedback: QAFeedback = Field(
        description="QA of the saved transcript (the final attempt)"
    )
    qa_cycles: int = Field(description="How many film + QA rounds ran")
    qa_history: list[QAAttempt] = Field(
        description="Scores from every attempt, in order"
    )
