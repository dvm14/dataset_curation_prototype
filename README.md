# Benchmark Transcript Generation Pipeline

Generates synthetic interdisciplinary research meeting transcripts for evaluating AI research assistants. Each transcript has a benchmark feature (seed) embedded at a known location with a ground truth label.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
ANTHROPIC_API_KEY=your_key_here
```

Add an AMI transcript to `data/ami_transcripts/sample.txt`. Any plain-text excerpt from the AMI Meeting Corpus works — used for style reference only, content is not reused.

## Run

```bash
python main.py
```

Output saved to `data/outputs/transcript_<id>.json`.

## Pipeline

```
Stage 1  Style extraction from AMI transcript
Agent 1  Content Curation   — research problem + disciplines + seed
Agent 2  Casting            — speaker profiles (randomized from value bank)
Agent 3  Filming            — generates the transcript
Agent 4  QA                 — LLM judge + one re-film cycle if needed
```

## Configuration

Edit the `CONFIG` dict in `main.py`:

| Key | Description |
|-----|-------------|
| `ami_transcript_path` | Path to AMI .txt file |
| `research_problem` | What the team is working on |
| `disciplines` | List of fields, or `None` to randomize |
| `discipline_distance` | `"close"` / `"distant"` / `"mixed"` |
| `benchmark_category` | Which category to test (jargon, assumptions, etc.) |
| `benchmark_task` | The evaluation task description |
| `speaker_overrides` | Optional per-discipline attribute overrides |

## Output schema

Each output JSON contains:
- Full transcript (ordered turns)
- Seed spec + ground truth (for evaluation)
- Speaker profiles
- QA scores
- AMI style notes used

## Value banks

Speaker attributes are randomized from `utils/value_banks.py` unless overridden. To fix an attribute for a discipline:

```python
"speaker_overrides": {
    "ecology": {"tone": "formal"},
    "economics": {"epistemic_stance": "skeptical of other disciplines' approaches"}
}
```
