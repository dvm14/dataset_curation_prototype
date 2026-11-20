"""Blind, transcript-only assumption identification pilot for one saved candidate."""

import argparse
import json
import os
from pathlib import Path

from utils.llm import call_llm_json


def run(path: Path, diagnostic: bool = False):
    sample = json.loads(path.read_text())
    qa = sample["qa_feedback"]
    if sample.get("benchmark_category") != "assumptions":
        raise ValueError("Pilot requires an assumptions candidate.")
    if not qa["approved"] and not (diagnostic and qa.get("seed_recoverable", 0) >= 4
                                    and qa.get("seed_implicit", 0) >= 4
                                    and qa.get("assumption_status") == "implicit"):
        raise ValueError("Pilot requires an approved candidate, or a seed-qualified diagnostic candidate with --diagnostic.")
    transcript = "\n".join(
        f"[{i}] {turn['speaker']}: {turn['text']}"
        for i, turn in enumerate(sample["transcript"], 1)
    )
    # The model input intentionally excludes seed, ground truth, briefs and QA.
    result = call_llm_json(
        """You are reading a cross-disciplinary research meeting. Identify one
unstated assumption that a shared research decision relies on. Explain the
decision and cite turn numbers from the transcript. Do not presume the
assumption is false. Return JSON with assumption, dependent_decision,
evidence_turns, explanation and risk_if_false.""",
        transcript,
    )
    output = {"transcript_id": sample["transcript_id"],
              "model_id": os.getenv("OPENAI_MODEL"), "pilot_result": result,
              "scope": ("transcript-only diagnostic on QA-rejected sample; not a qualifying pilot"
                        if not qa["approved"] else
                        "transcript-only pilot; not a validity or ability finding")}
    destination = path.with_name(f"{'diagnostic_' if not qa['approved'] else ''}pilot_{sample['transcript_id']}.json")
    destination.write_text(json.dumps(output, indent=2))
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("transcript_json", type=Path)
    parser.add_argument("--diagnostic", action="store_true",
                        help="Allow an implicit, seed-qualified sample rejected by other QA criteria")
    args = parser.parse_args()
    run(args.transcript_json, diagnostic=args.diagnostic)
