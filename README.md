# Benchmark Transcript Generation Pipeline

Generates synthetic interdisciplinary research meeting transcripts for evaluating AI research assistants. Each transcript is built for one task from Evaluation Benchmark v2 (`benchmark_tasks.py`) and carries the situation that task needs (the **seed**), for example one term used with two different disciplinary meanings, a timeline conflict nobody notices, or a clean control with no problem planted, plus a proposed reference answer (ground truth) the AI assistant's answer can be scored against.

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here     # only needed when a model below is a claude-* model
GEN_MODEL=gpt-5.6-terra             # writes the meetings (Agents 1-3, series planner)
JUDGE_MODEL=gpt-5.6-terra           # grades them (Agent 4 QA, series QA)
```

Any `claude-*` model goes to the Anthropic API; anything else to OpenAI. Both
roles default to `gpt-5.6-terra`. To generate with Claude but grade exactly as
before, set only `GEN_MODEL=claude-sonnet-5-5`: keeping one judge for every
generator makes their transcripts comparable. Claude calls use adaptive
thinking at `CLAUDE_EFFORT` (default `low`); thinking tokens are billed as output.

Add an AMI transcript to `data/ami_transcripts/sample.txt`. Any plain-text excerpt from the AMI Meeting Corpus works. It is used for style reference only; its content is not reused.

## Run

```bash
python main.py --list-tasks              # all 33 tasks, their input type and category
python main.py --task coordination_03 --dry-run   # show the task, no API calls
python main.py --task jargon_01          # generate one transcript
```

All 33 tasks have a runner. Single-meeting tasks retain the Lexi pipeline.
Multi-input tasks use Rita's `multi_meeting.py`, sharing the same filming and QA
functions, including Lexi's closing turns and exploration metrics.

```bash
python main.py --task integration_02 --dry-run  # inspect routing, no API calls
python main.py --task integration_02            # one sample with two meetings
```

| Task | Material in one sample |
| --- | --- |
| problem_02 | Research plan + one meeting |
| integration_02 | Two comparison meetings |
| self_review_01 | Three sequential meetings |
| adaptation_01 | Two sequential meetings |
| other_02 | One dialogue + two swapped seniority-label mappings |

Multi-input results are one JSON per sample, with a `meetings` list, overall
`ground_truth`, `series_qa`, and `approved`. They use the existing
`data/outputs/<task_id>/approved/` and `data/outputs/<task_id>/failed/` directories, created automatically when saving. Existing output files are not moved. For evaluation,
use only the intended meetings/plan/labels, not ground truth or QA metadata.

Existing Rita behavior is preserved: counts are fixed by input type; previous
meetings are carried forward as summaries; each meeting can be re-filmed, but
failed series-level QA does not automatically regenerate the series.
See `MERGE_NOTES.md` for the exact merge scope and validation limitations.

Output is saved to `data/outputs/approved/` or `data/outputs/failed/` as `<task_id>_<YYYY-MM-DD_HH-MM-SS>_<id>.json`. The script exits with code 2 when the transcript fails QA. At the end of each run it prints token usage and an estimated API cost.

#### Constraints enforced in code

Some task requirements are countable, and speakers kept "fixing" them, so the code
controls them (`utils/floor_control.py`):

- **Airtime** (`integration_01`): the dominant speaker takes the floor whenever they are
  below ~65% of the words; others get short turns; the quiet speaker stays out until
  turn 15, then makes their point, at most 3 turns in all.
- **Action items** (ground truth with `assigned_items` / `unassigned_items`, e.g.
  `coordination_05`): nobody reads out a list or adds a to-do during the meeting; the
  chair runs through exactly those items at the end, owners and dates checked in code
  and protected from the speech pass. For `coordination_05` only, a single
  `action_items` list is read too (item under `action`, `item`, `task` or
  `description`; no owner or `status: "unassigned"` means unassigned), since GPT's
  Agent 1 writes that format and the wrap-up was skipped without it.
- **Planned moments (beats)**: see [Agent 3](#agent-3-meeting-simulation), step 0.

## Batch runs

```bash
python run_batch.py --n 1                 # pilot: one sample for each of the 33 tasks
python run_batch.py --n 10 --workers 6    # ten per task
python run_batch.py --n 10 --budget 150   # stop starting new runs once $150 is spent
python run_batch.py --n 10 --dry-run      # what would run, no API calls
```

When `GEN_MODEL` is set, samples go to `data/outputs/<GEN_MODEL>/<task_id>/{approved,failed}/`
and record `generator_model` and `judge_model`, so runs from different generators
never mix. Samples already on disk count toward `--n`, so an interrupted batch
resumes where it stopped. Each sample runs as its own `main.py` process (log in
`<output root>/logs/`); a crashed run is retried once. A CSV summary is written at the end.

Sample *k* of a task runs as `main.py --seed k`, which fixes the research background,
meeting style, opening and speaker attributes. So sample *k* gets the same setup for
every generator model, and Claude and GPT outputs can be compared pair by pair.
Each saved JSON records `sample_seed`, `research_id`, `meeting_style` and `meeting_opening`.

To check a judge, re-score saved transcripts without regenerating them:

```bash
python rejudge.py --judge gpt-5.6-terra            # all single-meeting samples of GEN_MODEL
python rejudge.py --judge gpt-5.6-terra --limit 5
```

It writes `rejudge_<judge>_<time>.csv` with the saved and new scores side by side.

`RUN_TAG=v2` stores a run in `data/outputs/<GEN_MODEL>__v2/`, so a rerun on changed code
doesn't count earlier samples as done.

Offline test (fake LLM, no key or cost needed):

```bash
python -m pytest tests -q
python tests/test_beats.py        # beat parsing, scheduling, planting, value removal
```

**Results on this branch** (GPT-5.6-terra, one transcript per task, the 17 single-meeting tasks that failed in the 21 Nov batch): 14 of 17 approved on the first attempt, against 14 of 86 for the same tasks before. The judge changes don't explain it: the 18 pre-beats baseline transcripts re-scored with the current judge pass 2/18 (3/18 with the old one). After the hidden-conflict and action-item fixes, `coordination_03` passed; `assumptions_03` and `methods_03` still fail (see Agent 3, step 0), and `coordination_05` now runs its wrap-up but fails on decisions nothing makes sure are said and a minutes-like recap (naturalness 2).

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

0. **Planned moments (beats)** (`utils/beats.py`). In GPT runs most failures were the required content never being said, or said in a form the judge couldn't use, while speakers improvised freely. So for most tasks one call first plans the few moments the task can't do without, and the code places them:
   - **The plan.** 3–7 beats (at most 4 for summarising and planning tasks, `ANCHOR_TASKS`), each one point of at most 25 words with exact values, an **owner** (a cast speaker; people the seed names who weren't cast are left out), a **window** (early / middle / late), and optionally what must **never** happen after it. Decisions are beats too ("the group adopts X"). Meeting-wide **conditions** ("nobody notices that b1 and b4 conflict") are returned alongside.
   - **Scheduling.** Each beat gets a turn spread evenly within its window, at least 3 turns apart (`MIN_GAP`), and linked pieces of a hidden conflict at least 10 apart (`LINKED_GAP`). The quiet speaker's beats wait until they are allowed to speak (airtime tasks).
   - **Saying it.** When a beat's turn comes, its owner takes the floor (one turn later if they just spoke) and is told to bring the point up in passing, in their own words. A check call confirms the turn got it across; if not, the turn is regenerated once. The meeting can't end naturally while a beat is still due. Owners also see their points in their brief ("Points you plan to raise in this meeting").
   - **Leak check.** After a beat is said, its "never" joins the conditions the leak check enforces.
   - **Not used for** `jargon_01`, `jargon_02` (they keep the seed-term machinery) and `coordination_05` (the action-item wrap-up covers it) (`NO_BEAT_TASKS`). Those tasks keep the planner's required facts and nudges; beat tasks don't use them.
   - **Hidden-conflict tasks** (`assumptions_03`, `coordination_03`, `methods_03`, `HIDDEN_LINK_TASKS`). Once either piece of a conflict was known in advance, speakers connected the two as soon as the other came up. So:
     - the later piece is said live at its turn, by a different owner from the earlier piece;
     - the earlier piece is **planted** after the meeting as a short aside in one of its owner's earlier turns, at least 10 turns before, so no turn was written knowing it;
     - the planner doesn't see the seed instruction, Agent 1's ground truth or either piece, only the task description, research problem and the other beats;
     - any sentence with one of the pieces' exact values (dates either way round, clock times, numbers with a unit) is removed from Agent 1's research problem and meeting context, the shared background and every brief, and the removed sentences are printed;
     - a later piece may not repeat the earlier piece's values; if it does, the beats are planned once more.

     This held for `coordination_03` (dates). When Agent 1 picks a conflict of ideas instead (`assumptions_03`: an English-only survey for Spanish-speaking tenants; `methods_03`: a survey for a causal question) there are no values to remove and the participants still notice it, so those two still fail. For `methods_03`, removing "30 days" also removed the research problem's main sentence.
   - **Ground truth after the meeting.** Once the transcript is final, one call rewrites the reference answer from what was actually said (turn numbers, the cast speaker who said it), using Agent 1's draft and the beats. It is saved as `ground_truth`; Agent 1's draft stays in `seed_spec.ground_truth` and the differences are listed in `ground_truth_changes`.
   - **Log.** Each run prints the beats with their turns, `Beats: x/y conveyed`, and any beat never placed.

1. **Planning.** One call writes **shared background** that everyone in the meeting knows (city, institution, grant and deadline, datasets, what happened last meeting, logistics), plus a **private brief** per speaker. The task's required facts are split across briefs so the situation can arise, but nobody is told the answer or that they are being tested. For terminology tasks, seed speakers get their own meaning of the seed term as their everyday word for it. Each brief also gives the speaker things they don't know: one or two open questions (at least one about another participant's field), something they'd need to check, and a puzzling observation from their own work, with no ready-made solutions or task assignments unless the task calls for them. Without these, meetings were almost all proposals and volunteering.
2. **Turn-taking by bidding.** Before every turn, all speakers bid in parallel on how much they want to speak (urgency 0–10) and how: a normal response, a quick reaction, cutting in, or passing. The highest bid speaks, and ties are broken at random. Nobody assigns turns.
   - The same speaker can't take two turns in a row.
   - Terminology tasks only: when someone cuts in, the previous turn is actually cut off mid-sentence ("—"), never where that would remove the seed term. Other tasks don't cut, since a cut could drop a date or fact the task depends on.
   - From turn 15, the meeting ends on its own once every bid is below 3. Turn 35 is a hard stop. Either way, the meeting then gets a closing (step 7).
3. **Speaking.** The winner writes their turn following the speaking rules in `prompts/agent3_prompt.py`: spoken rather than written, exploring before settling (admit what you'd need to check, ask before solving, ask real questions of other fields, think out loud, don't volunteer for work after every point), mostly cooperative, disagreement only when genuine, no recap of action items. A turn far over its word budget is regenerated once, shorter.
4. **Listener reactions.** In the same bid, each speaker says whether they'd react out loud to the last turn ("mm-hm", "oh, really?", "wait, the 18th?") without taking the floor. Before the winner speaks, one such listener (not the last speaker or the winner) may get a reaction of up to about six words.
   - Let through 30% of the time (`REACTION_CHANCE`).
   - Nobody reacts twice without taking a real turn in between. With 3 speakers the eligible listener is often the same person, who otherwise ended up doing nothing but backchannel.
   - The reacting speaker sees the recent reactions and must not reuse their wording, and is told not to stamp approval ("exactly", "agreed").
5. **Time cue.** Agents are told how much of the 10-minute meeting has passed, based on words spoken (150 per minute), so any time they mention stays consistent. Near the turn cap they're told someone has a hard stop.
6. **Task safeguards.**
   - **Leak check:** a turn is checked for breaking a condition the task needs, for example resolving a conflict that must go unnoticed, assigning an owner who must stay unassigned, or adding confusion to a control. Explicit dates, definitions and decisions are fine when the task allows them. A violating turn is regenerated up to 2 times. The checker sees the last 8 turns (`LEAK_CONTEXT_TURNS`); the ground truth tells it what has to stay intact. If its reply is malformed, the turn is kept and QA judges it.
   - **Terminology tasks** (category Jargon, not a control; see `utils/seed_terms.py`): only turns containing the seed term are leak-checked, with extra rules against pointing out the two meanings, asking for a definition, switching to the other meaning, objecting to a label, or renaming one's own sense. Speakers are told not to define their field's terms or comment on others' word choice. From turn 8, a seed speaker who hasn't said the term yet gets a private nudge, and the meeting can't end naturally until every seed speaker has used it.
7. **Closing.** Ending the loop only stops the talking, so transcripts used to end on whatever substantive turn came last. Once it ends, the person who opened the meeting (or, if they gave the last turn, whoever spoke most apart from the last speaker) closes it in a sentence or two: why they're stopping, maybe one loose next step, and thanks. They don't recap, raise anything new, or assign tasks that weren't assigned in the meeting. The reason is picked at random from `CLOSING_STYLES` (out of time, covered what we needed, follow up by email, ...), because left alone almost every closing began "I think we're at time". A meeting that hit turn 35 always uses "someone has to leave now", matching the hard-stop time cue. Then 1 or 2 others say a short goodbye, each told what was already said so they don't repeat it. Closing turns go through the leak check (a leaking closing ends the meeting without it) and are not counted toward the 35 turns, so a meeting can reach 38.
8. **Speech pass.** After the meeting, every turn of 15+ words is rewritten in parallel to read like transcribed speech: no semicolons, colons or parentheses, partial lists, varied openers, same points and roughly the same length. Each turn gets one disfluency type picked at random in code (none, a restart, a corrected word, a repeated word, losing the thread, trailing off, searching for a word), because asking for a restart in every long turn produced the same "X- well, X" repair everywhere. A rewrite is discarded, and the original kept, if it changes the length outside 0.75–1.3×, fails the leak check, or (terminology tasks) changes how often the seed term is used. Turn off with `SPEECH_PASS = False`.
9. **Seed locator.** One call reports where the seed actually appears in the transcript.

Tuning constants (turn limits, word budgets, nudge timing, meeting clock) are at the top of `agents/agent3_filming.py`.

### Agent 4: QA rubric

The judge scores each criterion from 1 to 5, using 1/3/5 as anchors:

| Criterion | Needed to pass | What it checks |
|---|---|---|
| Seed recoverable | ≥ 4 | Evidence turns can be pointed to, and the ground truth follows from them |
| Task compliance | ≥ 4 | Every condition in the task description is met: required facts and roles present, issues unnoticed only where the task says so, controls free of the target problem |
| Naturalness | ≥ 3 | Realism signals present; no tells such as essay-like turns or manufactured conflict |
| Speaker distinctness | ≥ 3 | Speakers match their personas, with discipline-typical vocabulary |
| Domain alignment | ≥ 3 | Each speaker contributes from their own discipline (the research question itself may shift) |
| Dialogue flow | ≥ 3 | Turns follow on from each other |

The approval decision is made in code (`APPROVAL_THRESHOLDS` in `agents/agent4_qa.py`), not by the LLM. A missing or malformed score counts as 1. The judge also receives measured turn statistics (turn lengths, disfluency rate, turns per speaker) as context, which are not targets, and for terminology tasks, an exact count of which turns each speaker says the seed term in. It does not see the seed locator's report, which has misread speakers before.

**What the judge sees on beat runs.**
- The planned moments and conditions, and it scores seed recoverability and task compliance against them. Agent 1's seed is background: the judge doesn't fail a run because the seed named someone who wasn't cast or gave a part to a different speaker.
- The ground truth written after the meeting.
- The AMI style notes **without** the meeting style and opening. Shown those, the judge graded them as requirements ("not meaningfully tangent-heavy") even when told they were guidance.

**Meeting styles and outcome tasks.** Tasks whose meeting has to settle something (`OUTCOME_TASKS` in `benchmark_tasks.py`: `coordination_01`, `assumptions_01`, `problem_03`, `disagreement_02`, `methods_03`) never get the `thinking_aloud` or `tangent_heavy` style, which told speakers not to decide anything.

**Seed use check (terminology tasks).** A separate small call labels each turn containing the seed term (and the turn after it) with which discipline's sense it carries, and whether it comments on the word or renames a sense. In code, task_compliance is capped at 3 if any turn comments or renames, if nobody says the term, or if a seed speaker's sense never appears in a plain use. The main judge once gave 5 to a transcript whose only anthropology use objected to the label. The labels are saved as `seed_uses` (empty for other tasks).

**Spoken share.** The 1–5 naturalness score barely moves between versions of a transcript, so QA also labels each turn of 15+ words as sounding spoken or written and reports the share (`spoken_share`, with `written_turns`). It is for comparing runs and is not part of approval.

**Exploration share.** Meetings used to consist mostly of speakers turning each concern into a solution and volunteering to build it. QA labels each turn's main move (question, request, idea, observation, concern, answer, proposal, commitment, agreement, backchannel) and reports `exploration_share`, the share of non-backchannel turns that are a question, idea, observation or open concern, plus `turn_moves`, `volunteer_turns` (turns that offer or assign work) and `unsure_turns` (turns that admit not knowing). Like spoken share, it is for comparing runs and is not part of approval; two labellings of the same transcript agree within about 0.05. Transcripts from the first OpenAI runs scored 0.00–0.09; with the brief and speaking-prompt changes, about 0.2. The judge also has a matching naturalness tell.

## Configuration

Edit the `CONFIG` dict in `main.py`:

| Key | Description |
|-----|-------------|
| `ami_transcript_path` | Path to AMI .txt file |
| `research_problem` | What the team is working on |
| `disciplines` | List of fields, or `None` to use the task's default (`_DEFAULTS` in `benchmark_tasks.py`, e.g. `assumptions_02`) or randomize |
| `num_speakers` | Number of speakers / disciplines (default 3) |
| `discipline_distance` | `"close"` / `"distant"` / `"mixed"` |
| `task_id` | Which task from `benchmark_tasks.py` to generate (overridden by `--task`). Category and description come from the task table |
| `speaker_overrides` | Optional per-discipline attribute overrides |
| `max_refilms` | Re-films allowed if QA fails (default 1). Every attempt is QA'd |

## Output schema

Each output JSON contains:
- The transcript (ordered turns)
- Seed spec, ground truth, and the reported seed location. On beat runs `ground_truth` is the one written from the transcript, and `seed_spec.ground_truth` is Agent 1's draft
- `beats` (id, owner, window, says, linked, scheduled and actual turn, whether it got across, whether it was planted), `beat_conditions`, and `ground_truth_changes`; all empty for tasks without beats
- Speaker profiles
- `qa_feedback`: the QA of the saved transcript, including `evidence_turns`, `realism_signals` and `issues`
- `qa_history`: scores, approval and issues for every attempt, to see whether a re-film improved things
- `qa_cycles`: how many film + QA rounds ran (1 or 2)
- `generated_at` and the AMI style notes used

A transcript that still fails QA after the re-film is saved with `approved: false`, so it can be filtered out.

## Cost

All agents use `gpt-5.6-terra` with reasoning effort `none` (set by `MODEL` and `REASONING_EFFORT` in `utils/llm.py`). Agent 3 accounts for most of the cost: each turn makes 3 bid calls and a speaking call. Leak checks run on only the few turns with the seed term for terminology tasks, but on every turn and every speech-pass rewrite for other tasks (roughly two per turn), each with an 8-turn window. Speaker context and conversation history are prompt-cached with explicit cache breakpoints (`prompt_cache_breakpoint`), so each call pays full price mainly for the newest turn. Check the usage printout at the end of a run for actual costs. A run with a re-film costs roughly twice as much as one that passes on the first attempt.

## Value banks

Speaker attributes are randomized from `utils/value_banks.py` unless overridden. To fix an attribute for a discipline:

```python
"speaker_overrides": {
    "ecology": {"tone": "formal"},
    "economics": {"epistemic_stance": "skeptical of other disciplines' approaches"}
}
```
