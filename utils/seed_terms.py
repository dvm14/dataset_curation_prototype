"""
seed_terms.py
Exact-match helpers for terminology tasks (category "Jargon", not controls).

For these tasks the seed is one word used in two disciplinary senses, so
checks can be done in code instead of asking an LLM: whether the term was
said, by whom, and how often. Earlier runs showed why this matters: the
judge repeated the seed locator's false claim that a speaker never said
the term, and gave 5/5 to a transcript whose only use of one sense
objected to the label.

Every other task returns no terms here and runs without these checks.
"""

import json
import re


def is_term_task(benchmark_category, benchmark_task):
    """True when the task plants one shared term (jargon, not a control)."""
    return (
        benchmark_category.strip().lower() == "jargon"
        and not benchmark_task.strip().upper().startswith("CONTROL")
    )


def term_mode(content):
    return is_term_task(content.seed.benchmark_category, content.seed.benchmark_task)


def seed_terms(content):
    """
    The seed's key term(s), only for term-mode tasks. Agent 1 isn't
    consistent about the key ("term", "ambiguous_term", ...), so take any
    short string value under a key containing "term".
    """
    if not term_mode(content):
        return []

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

    walk(content.seed.ground_truth)
    return list(dict.fromkeys(terms))


def _pattern(term):
    """Whole-word match that also allows a plural ("networks")."""
    return rf"\b{re.escape(term)}(s|es)?\b"


def count_mentions(text, terms):
    return sum(
        len(re.findall(_pattern(term), text, re.IGNORECASE))
        for term in terms
    )


def mentions(text, terms):
    return any(re.search(_pattern(term), text, re.IGNORECASE) for term in terms)


def _stem(discipline):
    """Rough stem so "neuroscientist" matches "neuroscience"."""
    d = discipline.lower().strip()
    return d[:max(5, len(d) - 3)]


def seed_speakers(content, speakers):
    """
    Speakers the seed involves.

    Preferred: the disciplines listed in ground_truth["meanings"]. Name
    matching over the whole ground truth once picked up a third speaker
    who only appeared in a scripted example Agent 1 had put there.

    Fallback for older ground truths: speakers named (full or last name)
    in the seed spec, else speakers whose discipline appears in it.
    """
    meanings = content.seed.ground_truth.get("meanings")
    if isinstance(meanings, list):
        listed = [
            str(m.get("discipline", "")).lower().replace("_", " ")
            for m in meanings if isinstance(m, dict)
        ]
        matched = [
            speaker for speaker in speakers
            if any(_stem(speaker.discipline) in d or _stem(d) in speaker.discipline.lower()
                   for d in listed if d)
        ]
        if matched:
            return matched

    spec = content.seed.explicit_instruction + json.dumps(content.seed.ground_truth)
    named = [
        speaker for speaker in speakers
        if speaker.name in spec or speaker.name.split()[-1] in spec
    ]
    if named:
        return named

    truth = json.dumps(content.seed.ground_truth).lower().replace("_", " ")
    return [
        speaker for speaker in speakers
        if speaker.discipline.strip() and _stem(speaker.discipline) in truth
    ]


def term_uses(transcript, terms):
    """Measured, not judged: the turns where each speaker says the seed term."""
    uses = {}
    for number, turn in enumerate(transcript, start=1):
        if count_mentions(turn.text, terms):
            uses.setdefault(turn.speaker, []).append(number)
    if not uses:
        return f"Nobody says {terms}."
    return "\n".join(
        f"- {speaker}: turns {', '.join(map(str, numbers))}"
        for speaker, numbers in uses.items()
    )


def seed_nudge(terms):
    """Private reminder for a seed speaker who hasn't used the term yet."""
    return (
        "Something from your brief you haven't raised yet involves what you "
        f"call {terms[0]}. If it fits the conversation, bring it up now, "
        f"using the word \"{terms[0]}\" the way you normally would."
    )
