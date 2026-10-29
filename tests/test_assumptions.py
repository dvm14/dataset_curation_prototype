from schema import Turn
from agents.agent4_qa import validate_assumption_evidence
from agents.agent3_filming import seed_terms, cut_off
from agents.agent4_qa import run as qa_run
from schema import (ContentCurationOutput, SeedSpec, DisciplineDistance,
                    CastingOutput, SpeakerProfile, FunctionalRole, Tone,
                    LanguageComplexity, CommunicationStyle, EpistemicStance,
                    FilmingOutput)
import json
from pathlib import Path
from schema import GeneratedTranscript
from agents.agent1_content_curation import run as curate


def test_valid_evidence_uses_actual_turn_quotes():
    turns = [Turn(speaker="A", text="We will track cortisol after each visit."),
             Turn(speaker="B", text="Belonging and safety are part of wellbeing."),
             Turn(speaker="A", text="Then rank the parks by cortisol change for overall wellbeing.")]
    evidence = [
        {"turn": 1, "quote": "track cortisol", "role": "method"},
        {"turn": 2, "quote": "Belonging and safety", "role": "goal"},
        {"turn": 3, "quote": "rank the parks by cortisol change", "role": "decision"},
    ]
    assert validate_assumption_evidence(turns, evidence) == []


def test_missing_decision_evidence_fails():
    turns = [Turn(speaker="A", text="We might track cortisol."),
             Turn(speaker="B", text="We should also talk about belonging.")]
    evidence = [{"turn": 1, "quote": "track cortisol", "role": "method"},
                {"turn": 2, "quote": "talk about belonging", "role": "goal"}]
    assert "decision" in " ".join(validate_assumption_evidence(turns, evidence))


def test_bad_turn_and_quote_fail():
    turns = [Turn(speaker="A", text="We will measure cortisol.")]
    evidence = [{"turn": 4, "quote": "measure cortisol", "role": "method"},
                {"turn": 1, "quote": "invented", "role": "goal"},
                {"turn": 1, "quote": "measure cortisol", "role": "decision"}]
    assert len(validate_assumption_evidence(turns, evidence)) == 2


def test_assumption_words_are_not_jargon_seed_terms():
    truth = {"assumption": "Cortisol is enough", "dependent_decision": "rank parks"}
    assert seed_terms(truth) == []


def test_interruption_does_not_remove_assumption_evidence():
    turn = Turn(speaker="A", text="Well the samples look fine, and I think we should rank all parks by cortisol change for overall wellbeing next week")
    assert cut_off(turn, [], protect_full_turn=True) == turn


def make_qa_inputs(category="assumptions"):
    turns = [Turn(speaker="A", text="We will track cortisol."),
             Turn(speaker="B", text="Belonging matters for wellbeing."),
             Turn(speaker="A", text="Let's rank parks by cortisol for wellbeing.")]
    content = ContentCurationOutput(
        research_problem="parks", disciplines=["neuroscience", "anthropology"],
        discipline_distance=DisciplineDistance.DISTANT, meeting_context="study",
        seed=SeedSpec(benchmark_category=category, benchmark_task="identify",
                      explicit_instruction="method to decision", ground_truth={
                          "assumption": "cortisol suffices", "dependent_decision": "rank parks"}))
    profiles = [SpeakerProfile(name=name, discipline=field,
                functional_role=FunctionalRole.POSTDOC, background="researcher",
                distinct_perspective="field methods", tone=Tone.FORMAL,
                language_complexity=LanguageComplexity.PLAIN,
                communication_style=CommunicationStyle.COLLABORATIVE,
                epistemic_stance=EpistemicStance.OPEN)
                for name, field in [("A", "neuroscience"), ("B", "anthropology")]]
    return FilmingOutput(transcript=turns, seed_location="turns 1-3"), content, CastingOutput(speakers=profiles)


def test_explicit_or_resolved_assumption_is_not_approved(monkeypatch):
    import agents.agent4_qa as qa
    for status in ("exposed", "resolved"):
        raw = {key: 5 for key in qa.APPROVAL_THRESHOLDS}
        raw.update(assumption_status=status, inference="the measure is used for the broader goal",
                   evidence=[{"turn": 1, "quote": "track cortisol", "role": "method"},
                             {"turn": 2, "quote": "Belonging matters", "role": "goal"},
                             {"turn": 3, "quote": "rank parks", "role": "decision"}],
                   issues=[])
        monkeypatch.setattr(qa, "call_llm_json", lambda *args, **kwargs: raw)
        filming, content, casting = make_qa_inputs()
        result = qa_run(filming, content, casting, "natural")
        assert result.approved is False
        assert result.assumption_status == status


def test_generic_discussion_without_decision_is_rejected(monkeypatch):
    import agents.agent4_qa as qa
    raw = {key: 5 for key in qa.APPROVAL_THRESHOLDS}
    raw.update(assumption_status="missing", inference="no joint decision was made",
               evidence=[{"turn": 1, "quote": "track cortisol", "role": "method"},
                         {"turn": 2, "quote": "Belonging matters", "role": "goal"}],
               issues=[])
    monkeypatch.setattr(qa, "call_llm_json", lambda *args, **kwargs: raw)
    filming, content, casting = make_qa_inputs()
    result = qa_run(filming, content, casting, "natural")
    assert not result.approved
    assert any("decision" in issue for issue in result.issues)


def test_advisory_issue_does_not_change_thresholds(monkeypatch):
    import agents.agent4_qa as qa
    raw = {key: 5 for key in qa.APPROVAL_THRESHOLDS}
    raw.update(assumption_status="implicit", inference="the measure supports the choice",
               evidence=[{"turn": 1, "quote": "track cortisol", "role": "method"},
                         {"turn": 2, "quote": "Belonging matters", "role": "goal"},
                         {"turn": 3, "quote": "rank parks", "role": "decision"}],
               issues=["Minor conversational repetition."])
    monkeypatch.setattr(qa, "call_llm_json", lambda *args, **kwargs: raw)
    filming, content, casting = make_qa_inputs()
    assert qa_run(filming, content, casting, "natural").approved


def test_jargon_path_keeps_existing_approval_logic(monkeypatch):
    import agents.agent4_qa as qa
    raw = {key: 5 for key in qa.APPROVAL_THRESHOLDS}
    raw.update(evidence_turns=[1], issues=[])
    monkeypatch.setattr(qa, "call_llm_json", lambda *args, **kwargs: raw)
    filming, content, casting = make_qa_inputs("jargon")
    assert qa_run(filming, content, casting, "natural").approved is True


def test_historical_jargon_output_still_parses():
    path = next(Path("data/outputs").glob("transcript_2026-10-25*.json"))
    record = GeneratedTranscript.model_validate(json.loads(path.read_text()))
    assert record.benchmark_category == "jargon"


def test_curator_retries_blank_assumption_field(monkeypatch):
    import agents.agent1_content_curation as agent1
    calls = []
    def response(*args, **kwargs):
        calls.append(1)
        truth = {key: "valid" for key in agent1.ASSUMPTION_FIELDS}
        if len(calls) == 1:
            truth["why_needed"] = ""
        return {"seed": {"benchmark_category": "assumptions",
                         "benchmark_task": "task", "explicit_instruction": "seed",
                         "ground_truth": truth}}
    monkeypatch.setattr(agent1, "call_llm_json", response)
    result = curate("problem", ["a", "b"], DisciplineDistance.DISTANT,
                    "task", "assumptions")
    assert result.seed.ground_truth["why_needed"] == "valid"
    assert len(calls) == 2
