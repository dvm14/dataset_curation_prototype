"""
main.py
Benchmark transcript generation pipeline.
Wires Agents 1-4 in sequence and saves the final output.

Usage:
    python main.py

Configure the run parameters in the RUN CONFIG section below.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

from schema import DisciplineDistance, GeneratedTranscript
from utils.style_extractor import extract_style
from utils.value_banks import sample_disciplines
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
}


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(config: dict) -> GeneratedTranscript:
    transcript_id = str(uuid.uuid4())[:8]
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
    disciplines = config["disciplines"] or sample_disciplines(
        config["discipline_distance"].value
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
    print(f"      Cast {len(casting.speakers)} speakers: "
          f"{', '.join(s.name for s in casting.speakers)}")

    # ── Agent 3: Filming ────────────────────────
    print("[4/5] Agent 3 — Filming...")
    filming = agent3.run(
        content=content,
        casting=casting,
        ami_style_notes=ami_style_notes,
    )
    print(f"      Generated {len(filming.transcript)} turns")
    print(f"      Seed location: {filming.seed_location}")

    # ── Agent 4: QA ─────────────────────────────
    print("[5/5] Agent 4 — QA...")
    qa = agent4.run(
        filming=filming,
        content=content,
        casting=casting,
        ami_style_notes=ami_style_notes,
    )
    qa_cycles = 1
    if not qa.feedback.approved:
        print("      Re-film triggered.")
        qa_cycles = 2
        # use the revised transcript
        filming = qa.revised_transcript
    print(f"      Approved: {qa.feedback.approved} "
          f"(after {qa_cycles} cycle(s))")

    # ── Assemble final output ───────────────────
    final = GeneratedTranscript(
        transcript_id=transcript_id,
        generated_at=datetime.now().isoformat(),
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
        qa_feedback=qa.feedback,
        qa_cycles=qa_cycles,
    )

    # ── Save output ─────────────────────────────
    out_path = Path(f"data/outputs/transcript_{transcript_id}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final.model_dump(), f, indent=2)
    print(f"\n✓ Saved to {out_path}")

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
    print(f"QA scores — naturalness: {result.qa_feedback.conversational_naturalness}/5, "
          f"seed embedded: {result.qa_feedback.seed_embedded}, "
          f"approved: {result.qa_feedback.approved}")


if __name__ == "__main__":
    result = run_pipeline(CONFIG)
    print_transcript(result)
