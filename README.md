# Benchmark Transcript Generation Pipeline

Generates synthetic interdisciplinary research meeting transcripts for evaluating AI research assistants. Each transcript has a hidden benchmark feature (the **seed**), for example one term used with two different disciplinary meanings, plus a ground truth label the AI assistant's answer can be scored against.

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then set:
```
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=your_project_model_id
```
Choose a model enabled in your OpenAI API project that supports the Responses
API and JSON object mode. There is no implicit model default. All agents,
including the style extractor, bids, leak checks and QA, use this model.
The repository `.env` is loaded regardless of the working directory; exported
environment variables take precedence. Keep the key out of Git.

For reasoning models, optionally set `OPENAI_REASONING_EFFORT` to a value
supported by that model. Reasoning and visible text share `max_output_tokens`:
short bid calls inherited from the prototype may need a larger budget. Set
`OPENAI_MIN_OUTPUT_TOKENS` (for example, `2048`) to give every call a larger
minimum budget. Incomplete output fails explicitly; it is never accepted as a
valid transcript. These settings require verification on your chosen model.

Add an AMI transcript to `data/ami_transcripts/sample.txt`. Any plain-text excerpt from the AMI Meeting Corpus works. It is used for style reference only; its content is not reused.

## Run

```bash
python main.py
```

Output is saved to `data/outputs/transcript_<YYYY-MM-DD_HH-MM-SS>_<id>.json`. At the end of each run, the script prints cumulative session token usage.
Dollar estimates are omitted; use your OpenAI project billing for actual costs.

## Pipeline

```
Stage 1  Style extraction   — style notes from the AMI transcript
Agent 1  Content Curation   — research problem, disciplines, seed + ground truth
Agent 2  Casting            — one speaker profile per discipline (3 by default)
Agent 3  Filming            — simulates the meeting, one agent per speaker
Agent 4  QA                 — scores the transcript against the rubric
         Re-film loop       — if QA fails, re-film once with QA's issues, then QA again
```

### Agent 3: meeting simulation

No single LLM call writes the transcript. Each cast speaker is a separate agent that sees only its own profile and private brief, plus what has been said so far.

1. **Planning.** One call writes **shared background** that everyone in the meeting knows (city, institution, grant and deadline, datasets, what happened last meeting, logistics), plus a **private brief** per speaker. Seed speakers get their own meaning of the seed term as their everyday word for it. Nobody is told that someone else uses it differently.
2. **Turn-taking by bidding.** Before every turn, all speakers bid in parallel on how much they want to speak (urgency 0–10) and how: a normal response, a quick reaction, cutting in, or passing. The highest bid speaks, and ties are broken at random. Nobody assigns turns.
   - The same speaker can't take two turns in a row.
   - When someone cuts in, the previous turn is actually cut off mid-sentence ("—"). It is never cut where that would remove the seed term.
   - From turn 15, the meeting ends on its own once every bid is below 3. Turn 35 is a hard stop.
3. **Speaking.** The winner writes their turn following the speaking rules in `prompts/agent3_prompt.py`: spoken rather than written, mostly cooperative, disagreement only when genuine, no recap of action items. A turn far over its word budget is regenerated once, shorter.
4. **Time cue.** Agents are told how much of the 10-minute meeting has passed, based on words spoken (150 per minute), so any time they mention stays consistent. Near the turn cap they're told someone has a hard stop.
5. **Seed safeguards.**
   - **Leak check:** each turn containing the seed term is checked for giving the seed away (pointing out the two meanings, asking for a definition, silently switching to the other meaning). A leaking turn is regenerated up to 2 times.
   - **Coverage:** from turn 8, a seed speaker who hasn't said the term yet gets a private nudge to bring it up. The meeting can't end naturally until every seed speaker has used it.
6. **Seed locator.** After the meeting, one call reports where the seed actually appears in the transcript.

Tuning constants (turn limits, word budgets, nudge timing, meeting clock) are at the top of `agents/agent3_filming.py`.

### Agent 4: QA rubric

The judge scores each criterion from 1 to 5, using 1/3/5 as anchors:

| Criterion | Needed to pass | What it checks |
|---|---|---|
| Seed recoverable | ≥ 4 | Evidence turns can be pointed to, and the ground truth follows from them |
| Seed implicit | ≥ 4 | Nobody names the issue, asks about it, or silently adjusts their usage |
| Naturalness | ≥ 4 | Realism signals present; no tells such as essay-like turns or manufactured conflict |
| Speaker distinctness | ≥ 3 | Speakers match their personas, with discipline-typical vocabulary |
| Topical alignment | ≥ 3 | Stays on the research problem |
| Dialogue flow | ≥ 3 | Turns follow on from each other |

The approval decision is made in code (`APPROVAL_THRESHOLDS` in `agents/agent4_qa.py`), not by the LLM. A missing or malformed score counts as 1. The judge also receives measured turn statistics (turn lengths, disfluency rate, turns per speaker) as context. These are not targets.

## Configuration

Edit the `CONFIG` dict in `main.py`:

| Key | Description |
|-----|-------------|
| `ami_transcript_path` | Path to AMI .txt file |
| `research_problem` | What the team is working on |
| `disciplines` | List of fields, or `None` to randomize |
| `num_speakers` | Number of speakers / disciplines (default 3) |
| `discipline_distance` | `"close"` / `"distant"` / `"mixed"` |
| `benchmark_category` | Which category to test (jargon, assumptions, etc.) |
| `benchmark_task` | The evaluation task description |
| `speaker_overrides` | Optional per-discipline attribute overrides |
| `max_refilms` | Re-films allowed if QA fails (default 1). Every attempt is QA'd |

## Output schema

Each output JSON contains:
- The transcript (ordered turns)
- Seed spec, ground truth, and the reported seed location
- Speaker profiles
- `qa_feedback`: the QA of the saved transcript, including `evidence_turns`, `realism_signals` and `issues`
- `qa_history`: scores, approval and issues for every attempt, to see whether a re-film improved things
- `qa_cycles`: how many film + QA rounds ran (1 or 2)
- `generated_at` and the AMI style notes used

A transcript that still fails QA after the re-film is saved with `approved: false`, so it can be filtered out.

## Cost

All agents use OpenAI through `utils/llm.py`. Agent 3 accounts for most calls:
each turn makes a bid call per speaker, a speaking call, and sometimes a leak
check. Repeated prefixes can benefit from OpenAI automatic prompt caching;
there are no Anthropic cache directives. The usage report counts total input,
cached input (a subset of input), and output tokens, aggregated safely across
parallel bids. Refused or incomplete responses still count toward usage.

JSON calls use Responses API JSON object mode; field-level validation remains
in the agents' Pydantic models. JSON mode does not guarantee field correctness.
No server conversation state is shared between participants (`store=False`).

API reference: [JSON output handling](https://developers.openai.com/api/docs/guides/structured-outputs).

## Offline tests and live check

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests use the real OpenAI SDK with an in-memory HTTP transport; no credentials
or paid API calls are required. For a manual live check after configuring `.env`:

```bash
python test_style_extractor.py
```

This command uses API credits. Run `python main.py` afterwards for the full
meeting pipeline. Existing saved transcripts are historical Claude outputs;
they have not been regenerated or revalidated on OpenAI.

## Value banks

## Assumptions pilot

Use the repository virtual environment and configured `.env`:

```bash
OPENAI_MIN_OUTPUT_TOKENS=2048 .venv/bin/python main.py --config configs/assumptions-green-space.json
OPENAI_MIN_OUTPUT_TOKENS=2048 .venv/bin/python main.py --config configs/assumptions-heat.json
OPENAI_MIN_OUTPUT_TOKENS=2048 .venv/bin/python main.py --config configs/assumptions-learning.json
```

Each command writes a JSON record in `data/outputs/`, including failures after
the configured QA re-films. `qa_feedback.approved` is the outcome. Assumptions
records include exact turn quotes, an inference, an implicit/exposed/resolved/
missing status, model ID, run configuration, and the transcript and QA for
each filming attempt. Quote matching and turn bounds are checked in code;
the meaning of the evidence requires human review. Historical jargon JSON
records remain readable with the expanded schema. The built-in jargon run is
still `.venv/bin/python main.py`.

For a transcript-only identification pilot on an approved candidate:

```bash
.venv/bin/python pilot_assumption.py data/outputs/transcript_<timestamp>_<id>.json
```

This pilot excludes the seed and ground truth from the identification request.
Generation, QA and the pilot may use the same model, so pilot agreement is not
independent validation. The assumptions category is an implementation proposal,
not a frozen team benchmark specification.

## Value banks

Speaker attributes are randomized from `utils/value_banks.py` unless overridden. To fix an attribute for a discipline:

```python
"speaker_overrides": {
    "ecology": {"tone": "formal"},
    "economics": {"epistemic_stance": "skeptical of other disciplines' approaches"}
}
```
