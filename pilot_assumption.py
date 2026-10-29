"""Blind, transcript-only assumption identification pilot for one saved candidate."""

import argparse
import json
import os
from pathlib import Path

from utils.llm import call_llm_json


def run(path: Path):
    sample = json.loads(path.read_text())
    if sample.get("benchmark_category") != "assumptions" or not sample["qa_feedback"]["approved"]:
        raise ValueError("Pilot requires an approved assumptions candidate.")
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
              "scope": "transcript-only pilot; not a validity or ability finding"}
    destination = path.with_name(f"pilot_{sample['transcript_id']}.json")
    destination.write_text(json.dumps(output, indent=2))
    print(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("transcript_json", type=Path)
    run(parser.parse_args().transcript_json)
