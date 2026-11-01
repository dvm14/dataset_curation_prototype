# Benchmark Transcript Generation Pipeline

Generates synthetic interdisciplinary research meeting transcripts for evaluating AI research assistants. Each transcript has a hidden benchmark feature (the **seed**), for example one term used with two different disciplinary meanings, plus a ground truth label the AI assistant's answer can be scored against.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
OPENAI_API_KEY=your_key_here
```

Add an AMI transcript to `data/ami_transcripts/sample.txt`. Any plain-text excerpt from the AMI Meeting Corpus works. It is used for style reference only; its content is not reused.

## Run

```bash
python main.py
```

Output is saved to `data/outputs/transcript_<YYYY-MM-DD_HH-MM-SS>_<id>.json`. At the end of each run, the script prints token usage and an estimated API cost.

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
4. **Listener reactions.** In the same bid, each speaker says whether they'd react out loud to the last turn ("mm-hm", "oh, really?", "wait, the 18th?") without taking the floor. Before the winner speaks, one such listener (not the last speaker or the winner) may get a reaction of up to about six words.
   - Let through 30% of the time (`REACTION_CHANCE`).
   - Nobody reacts twice without taking a real turn in between. With 3 speakers the eligible listener is often the same person, who otherwise ended up doing nothing but backchannel.
   - The reacting speaker sees the recent reactions and must not reuse their wording, and is told not to stamp approval ("exactly", "agreed").
5. **Time cue.** Agents are told how much of the 10-minute meeting has passed, based on words spoken (150 per minute), so any time they mention stays consistent. Near the turn cap they're told someone has a hard stop.
6. **Seed safeguards.**
   - **Leak check:** each turn containing the seed term is checked for giving the seed away (pointing out the two meanings, asking for a definition, silently switching to the other meaning, objecting to what something is called, or renaming one's own sense). A leaking turn is regenerated up to 2 times.
   - **Coverage:** from turn 8, a seed speaker who hasn't said the term yet gets a private nudge to bring it up. The meeting can't end naturally until every seed speaker has used it.
7. **Speech pass.** After the meeting, every turn of 15+ words is rewritten in parallel to read like transcribed speech: no semicolons, colons or parentheses, partial lists, varied openers, same points and roughly the same length. Each turn gets one disfluency type picked at random in code (none, a restart, a corrected word, a repeated word, losing the thread, trailing off, searching for a word), because asking for a restart in every long turn produced the same "X- well, X" repair everywhere. A rewrite is discarded, and the original kept, if it changes how often the seed term is used, changes the length outside 0.75–1.3×, or fails the leak check. Turn off with `SPEECH_PASS = False`.
8. **Seed locator.** One call reports where the seed actually appears in the transcript.

Tuning constants (turn limits, word budgets, nudge timing, meeting clock) are at the top of `agents/agent3_filming.py`.

### Agent 4: QA rubric

The judge scores each criterion from 1 to 5, using 1/3/5 as anchors:

| Criterion | Needed to pass | What it checks |
|---|---|---|
| Seed recoverable | ≥ 4 | Evidence turns can be pointed to, and the ground truth follows from them |
| Seed implicit | ≥ 4 | Nobody names the issue, asks about it, or silently adjusts their usage |
| Naturalness | ≥ 3 | Realism signals present; no tells such as essay-like turns or manufactured conflict |
| Speaker distinctness | ≥ 3 | Speakers match their personas, with discipline-typical vocabulary |
| Domain alignment | ≥ 3 | Each speaker contributes from their own discipline (the research question itself may shift) |
| Dialogue flow | ≥ 3 | Turns follow on from each other |

The approval decision is made in code (`APPROVAL_THRESHOLDS` in `agents/agent4_qa.py`), not by the LLM. A missing or malformed score counts as 1. The judge also receives measured turn statistics (turn lengths, disfluency rate, turns per speaker) as context, which are not targets, and an exact count of which turns each speaker says the seed term in. It does not see the seed locator's report, which has misread speakers before.

**Seed use check.** A separate small call labels each turn containing the seed term (and the turn after it) with which discipline's sense it carries, and whether it comments on the word or renames a sense. In code, seed_implicit is capped at 3 if any turn comments or renames, or if a seed speaker's sense never appears in a plain use. The main judge once gave 5 to a transcript whose only anthropology use objected to the label. The labels are saved as `seed_uses`.

**Spoken share.** The 1–5 naturalness score barely moves between versions of a transcript, so QA also labels each turn of 15+ words as sounding spoken or written and reports the share (`spoken_share`, with `written_turns`). It is for comparing runs and is not part of approval.

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

All agents use `gpt-5.6-terra` with reasoning effort `none` (set by `MODEL` and `REASONING_EFFORT` in `utils/llm.py`). Agent 3 accounts for most of the cost: each turn makes 3 bid calls, a speaking call and sometimes a leak check. Speaker context and conversation history are prompt-cached with explicit cache breakpoints (`prompt_cache_breakpoint`), so each call pays full price mainly for the newest turn. Check the usage printout at the end of a run for actual costs. A run with a re-film costs roughly twice as much as one that passes on the first attempt.

## Value banks

Speaker attributes are randomized from `utils/value_banks.py` unless overridden. To fix an attribute for a discipline:

```python
"speaker_overrides": {
    "ecology": {"tone": "formal"},
    "economics": {"epistemic_stance": "skeptical of other disciplines' approaches"}
}
```
