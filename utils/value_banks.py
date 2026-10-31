"""
value_banks.py
Default value banks for speaker attributes.
The Casting agent draws from these randomly unless a value is specified.
"""

import random
from schema import (
    FunctionalRole, Tone, LanguageComplexity,
    CommunicationStyle, EpistemicStance
)

# ─────────────────────────────────────────────
# ATTRIBUTE BANKS
# ─────────────────────────────────────────────

FUNCTIONAL_ROLES = list(FunctionalRole)
TONES = list(Tone)
LANGUAGE_COMPLEXITIES = list(LanguageComplexity)
COMMUNICATION_STYLES = list(CommunicationStyle)
EPISTEMIC_STANCES = list(EpistemicStance)

# Discipline pairings by distance
DISCIPLINE_PAIRS = {
    "distant": [
        ["ecology", "economics"],
        ["neuroscience", "political science"],
        ["biomedical engineering", "anthropology"],
        ["computer science", "education"],
        ["epidemiology", "science & technology studies"],
        ["ecology", "anthropology"],
        ["neuroscience", "economics"],
        ["computer science", "sociology"],
    ],
    "close": [
        ["ecology", "biology"],
        ["computer science", "biomedical engineering"],
        ["epidemiology", "public health"],
        ["neuroscience", "biology"],
        ["economics", "political science"],
    ],
    "mixed": [
        ["ecology", "economics", "anthropology"],
        ["computer science", "epidemiology", "sociology"],
        ["neuroscience", "education", "biomedical engineering"],
        ["political science", "economics", "science & technology studies"],
    ]
}


def sample_speaker_attributes(overrides: dict = {}) -> dict:
    """
    Returns a random set of speaker attributes.
    Any key in overrides replaces the random value.
    
    Usage:
        attrs = sample_speaker_attributes()
        attrs = sample_speaker_attributes({"tone": Tone.FORMAL})
    """
    defaults = {
        "functional_role":     random.choice(FUNCTIONAL_ROLES),
        "tone":                random.choice(TONES),
        "language_complexity": random.choice(LANGUAGE_COMPLEXITIES),
        "communication_style": random.choice(COMMUNICATION_STYLES),
        "epistemic_stance":    random.choice(EPISTEMIC_STANCES),
    }
    return {**defaults, **overrides}


def sample_disciplines(distance: str, n: int = None) -> list[str]:
    """
    Returns a list of disciplines for the given distance setting.
    n overrides the default pair/group size.
    
    Usage:
        disciplines = sample_disciplines("distant")
        disciplines = sample_disciplines("mixed", n=4)
    """
    options = DISCIPLINE_PAIRS.get(distance, DISCIPLINE_PAIRS["distant"])
    group = list(random.choice(options))  # copy so the bank isn't mutated
    if n:
        # if more disciplines requested than in group, sample with replacement
        while len(group) < n:
            extra = random.choice(DISCIPLINE_PAIRS[distance])
            for d in extra:
                if d not in group:
                    group.append(d)
    return group[:n] if n else group
