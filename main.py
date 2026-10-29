"""
main.py
Benchmark transcript generation pipeline.
Wires Agents 1-4 in sequence and saves the final output.

Usage:
    python main.py

Configure the run parameters in the RUN CONFIG section below.
"""

import json
import os
import argparse
import uuid
from datetime import datetime
from pathlib import Path

from schema import DisciplineDistance, GeneratedTranscript, QAAttempt
from agents.agent4_qa import APPROVAL_THRESHOLDS
from utils.style_extractor import extract_style
from utils.value_banks import sample_disciplines
from utils.llm import usage_report
import agents.agent1_content_curation as agent1
import agents.agent2_casting as agent2
import agents.agent3_filming as agent3
import agents.agent4_qa as agent4


# ─────────────────────────────────────────────
# RUN CONFIG — edit this for each run
# ─────────────────────────────────────────────

CONFIG = {
    # Path to your AMI transcript file
    "ami_transcript_path": "data/ami_transcripts/sample.txt",

    # Research setup — set manually or leave disciplines=None to randomize
    "research_problem": (
        "Understanding how urban green space affects mental health outcomes "
        "in low-income communities"
    ),
    "disciplines": None,  # None = randomize from value bank
    "num_speakers": 3,    # one cast speaker per discipline
    "discipline_distance": DisciplineDistance.DISTANT,

    # Benchmark task to embed
    "benchmark_category": "jargon",
    "benchmark_task": (
        "Given a meeting transcript where the same term was used by two "
        "different researchers from two different disciplines, test that the "
        "AI identifies the term as having two different meanings with "
        "definitions according to each discipline."
    ),

    # Optional: override specific speaker attributes by discipline
    # e.g. {"ecology": {"tone": "formal"}}
    "speaker_overrides": {},

    # Re-film at most this many times if QA fails. Every attempt is QA'd.
    "max_refilms": 1,
}


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(config: dict) -> GeneratedTranscript:
    config = dict(config)
    if isinstance(config.get("discipline_distance"), str):
        config["discipline_distance"] = DisciplineDistance(config["discipline_distance"])
    if config.get("benchmark_category") not in {"jargon", "assumptions"}:
        raise ValueError("benchmark_category must be jargon or assumptions")
    transcript_id = str(uuid.uuid4())[:8]
    created_at = datetime.now()
    print(f"\n{'='*50}")
    print(f"Pipeline run: {transcript_id}")
    print(f"{'='*50}\n")

    # ── Stage 1: extract AMI style ──────────────
    print("[1/5] Extracting AMI style...")
    ami_path = Path(config["ami_transcript_path"])
    if not ami_path.exists():
        raise FileNotFoundError(
            f"AMI transcript not found at {ami_path}. "
            "Add a .txt file from the AMI dataset to data/ami_transcripts/"
        )
    ami_raw = ami_path.read_text()
    ami_style_notes = extract_style(ami_raw)
    print(f"      Style notes extracted ({len(ami_style_notes)} chars)")

    # ── Agent 1: Content Curation ───────────────
    print("[2/5] Agent 1 — Content Curation...")
    num_speakers = config["num_speakers"]
    disciplines = config["disciplines"] or sample_disciplines(
        config["discipline_distance"].value, n=num_speakers
    )
    if len(disciplines) != num_speakers:
        raise ValueError(
            f"Expected {num_speakers} disciplines, got {len(disciplines)}: "
            f"{disciplines}"
        )
    content = agent1.run(
        research_problem=config["research_problem"],
        disciplines=disciplines,
        discipline_distance=config["discipline_distance"],
        benchmark_task=config["benchmark_task"],
        benchmark_category=config["benchmark_category"],
    )
    print(f"      Disciplines: {', '.join(content.disciplines)}")
    print(f"      Seed: {content.seed.explicit_instruction[:100]}...")

    # ── Agent 2: Casting ────────────────────────
    print("[3/5] Agent 2 — Casting...")
    casting = agent2.run(
        content=content,
        speaker_overrides=config.get("speaker_overrides", {}),
    )
    if len(casting.speakers) != num_speakers:
        raise ValueError(
            f"Expected {num_speakers} speakers, casting returned "
            f"{len(casting.speakers)}"
        )
    print(f"      Cast {len(casting.speakers)} speakers: "
          f"{', '.join(s.name for s in casting.speakers)}")

    # ── Agents 3 + 4: Filming and QA ────────────
    # Film, QA, and re-film with QA's issues if it fails.
    # Every attempt is QA'd, so the saved scores match the saved transcript.
    qa_history = []
    qa_feedback_text = None
    for attempt in range(1, config["max_refilms"] + 2):
        label = "Filming" if attempt == 1 else f"Re-filming (attempt {attempt})"
        print(f"[4/5] Agent 3 — {label}...")
        filming = agent3.run(
            content=content,
            casting=casting,
            ami_style_notes=ami_style_notes,
            qa_feedback=qa_feedback_text,
        )
        print(f"      Generated {len(filming.transcript)} turns")
        print(f"      Seed location: {filming.seed_location}")

        print(f"[5/5] Agent 4 — QA (attempt {attempt})...")
        qa = agent4.run(
            filming=filming,
            content=content,
            casting=casting,
            ami_style_notes=ami_style_notes,
        )
        scores = {key: getattr(qa, key) for key in APPROVAL_THRESHOLDS}
        qa_history.append(QAAttempt(
            attempt=attempt,
            num_turns=len(filming.transcript),
            scores=scores,
            approved=qa.approved,
            issues=qa.issues,
            transcript=filming.transcript,
            evidence=qa.evidence,
            assumption_status=qa.assumption_status,
        ))
        print(f"      Scores: {scores}")
        print(f"      Approved: {qa.approved}")

        if qa.approved:
            break
        qa_feedback_text = "\n".join(f"- {issue}" for issue in qa.issues)

    qa_cycles = len(qa_history)

    # ── Assemble final output ───────────────────
    final = GeneratedTranscript(
        transcript_id=transcript_id,
        generated_at=created_at.isoformat(),
        ami_style_notes=ami_style_notes,
        research_problem=content.research_problem,
        disciplines=content.disciplines,
        discipline_distance=content.discipline_distance,
        speakers=casting.speakers,
        benchmark_category=content.seed.benchmark_category,
        benchmark_task=content.seed.benchmark_task,
        seed_spec=content.seed,
        seed_location=filming.seed_location,
        ground_truth=content.seed.ground_truth,
        transcript=filming.transcript,
        qa_feedback=qa,
        qa_cycles=qa_cycles,
        qa_history=qa_history,
        model_id=os.getenv("OPENAI_MODEL"),
        run_config={**{key: (value.value if isinstance(value, DisciplineDistance) else value)
                       for key, value in config.items()},
                    "openai_reasoning_effort": os.getenv("OPENAI_REASONING_EFFORT", ""),
                    "openai_min_output_tokens": os.getenv("OPENAI_MIN_OUTPUT_TOKENS", "0")},
    )

    # ── Save output ─────────────────────────────
    timestamp = created_at.strftime("%Y-%m-%d_%H-%M-%S")
    out_path = Path(f"data/outputs/transcript_{timestamp}_{transcript_id}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final.model_dump(), f, indent=2)
    print(f"\n✓ Saved to {out_path}")
    print(f"\nAPI usage this session:\n{usage_report()}")

    return final


def print_transcript(result: GeneratedTranscript):
    print(f"\n{'─'*50}")
    print(f"TRANSCRIPT [{result.transcript_id}]")
    print(f"Topic: {result.research_problem}")
    print(f"Disciplines: {', '.join(result.disciplines)}")
    print(f"{'─'*50}\n")
    for turn in result.transcript:
        print(f"{turn.speaker}: {turn.text}\n")
    print(f"{'─'*50}")
    print(f"Seed location: {result.seed_location}")
    print(f"Ground truth: {json.dumps(result.ground_truth, indent=2)}")
    qa = result.qa_feedback
    print(f"QA scores — seed recoverable: {qa.seed_recoverable}/5, "
          f"seed implicit: {qa.seed_implicit}/5, "
          f"naturalness: {qa.naturalness}/5, "
          f"speaker distinctness: {qa.speaker_distinctness}/5, "
          f"approved: {qa.approved}")
    if len(result.qa_history) > 1:
        print("QA history:")
        for attempt in result.qa_history:
            scores = ", ".join(f"{k} {v}" for k, v in attempt.scores.items())
            print(f"  attempt {attempt.attempt}: {scores} "
                  f"→ approved: {attempt.approved}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="JSON configuration file; default is built-in jargon setup")
    args = parser.parse_args()
    run_config = json.loads(Path(args.config).read_text()) if args.config else CONFIG
    result = run_pipeline(run_config)
    print_transcript(result)
