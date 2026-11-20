# Lexi base + Rita multi-meeting merge

## Scope

Base: dataset_curation_prototype-lexi-meeting-simulation (2)(1).zip.
Merged source: dataset_curation_prototype-rita-multi-meeting.zip.

Lexi's closing/goodbye function and styles, exploratory private briefs,
exploration-oriented speaking prompts, speech pass, QA move classification,
exploration metrics and data schema are retained. The QA Python implementation,
schema, benchmark definitions and API utility are byte-for-byte unchanged.
Original Lexi example outputs are retained; they are not newly generated samples.

## Existing files changed

- main.py: extract shared AMI loading and film/QA helpers; route multi-input
  tasks to Rita's runner; keep Lexi exploration metrics in QA history and logs.
- agents/agent2_casting.py: import Rita's one-speaker-per-discipline handling.
- agents/agent3_filming.py: add Rita required-fact tracking and speech-pass
  protection while retaining all Lexi closing and exploration behavior.
- prompts/agent1_prompt.py: import the bounded required-facts guidance.
- prompts/agent3_prompt.py: add the new-turn-only leak-check guidance;
  preserve Lexi exploratory speaking instructions.
- prompts/agent4_prompt.py: add task-focused vocabulary/definition guidance;
  retain Lexi naturalness and move-check prompts.
- tests/test_task_pipeline.py: merge Rita test cases and restore the fake
  model handler for Lexi's move classification.
- README.md: document the multi-meeting entry point and output shape.

## New files

- multi_meeting.py: Rita runner, with benchmark-specific output folders.
- prompts/series_prompt.py: copied from Rita unchanged.
- utils/required_facts.py: copied from Rita unchanged.
- tests/test_multi_meeting.py, tests/__init__.py: copied from Rita unchanged.
- tests/test_merge_regression.py: closing and exploration in multi-meeting
  runs, plus required-date preservation during speech rewriting.
- MERGE_NOTES.md: this document.

## Run

Use your existing local environment and .env configuration. The ZIP contains
no API credentials. Install dependencies if needed:

```bash
python -m pip install -r requirements.txt
python main.py --task integration_02 --dry-run
python main.py --task integration_02
```

This produces one sample containing two meetings in a single JSON file.
The existing CLI is retained; no --samples or --count option was added.

## Validation

24 offline tests passed using fake model responses, including all five
multi-input types, single-meeting behavior, series QA rejection, shared
characters, context summaries, closing, exploration metrics and fact tracking.
No paid API calls or real-model generation were performed. Offline tests
validate integration, not the quality or benchmark validity of generated text.

## Retained limitations

For a minimal merge, Rita's input-type counts and series behavior are unchanged.
Series QA failures are saved, not automatically re-generated as a group.
Earlier meetings are summarized rather than passed in full. Required-fact
matching checks cues rather than semantic correctness and can miss errors.
The casting fallback can relabel an unmatched profile; review generated
speaker backgrounds for domain consistency. Model/API settings remain Lexi's.

## Output folders update

New single-meeting and multi-meeting samples are saved under
`data/outputs/<task_id>/approved/` or `data/outputs/<task_id>/failed/`.
Folders are created on demand. Existing outputs are not moved.
Each multi-meeting sample remains one JSON containing all its meetings.
Ten samples therefore means ten JSON files, which may contain more than ten
individual meeting transcripts. This update does not add batch-count CLI flags.
