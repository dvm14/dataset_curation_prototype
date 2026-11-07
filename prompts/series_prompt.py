"""
series_prompt.py
Prompts for multi-input tasks (multi_meeting.py): planning a set of linked
meetings, summarising a meeting for the next one, and judging the set.
"""

SERIES_PLAN_PROMPT = """
You design the material for one benchmark sample that needs more than one
meeting transcript (or a plan document plus a meeting). An AI research
assistant will later be given all of it and asked to do the benchmark task.

Plan the set so the task is answerable from the material alone. Each meeting
is simulated separately by speaker agents, so give each meeting:
- "label": short, with a date (e.g. "Meeting 2 — Feb 9, 2027")
- "meeting_context": what this meeting is for, as the team would describe it.
  Do not mention testing or what the evaluated AI should find.
- "instruction": what must happen in this meeting, for the simulator: who
  raises what, which facts are stated, what stays unconnected or unresolved.
  Only this meeting's part; facts from earlier meetings are carried over by
  the team's memory of them, not repeated here.
- "ground_truth": what this one transcript must contain, so it can be
  checked on its own (facts, positions, decisions). Values only, no dialogue.

Use concrete, consistent details across meetings: same project, same people
(unless the task needs different groups), dates in order, numbers that agree.
Never include sample dialogue anywhere.

Also return "ground_truth" for the whole set: the reference answer to the
benchmark task, citing which meeting each point comes from.

STRUCTURE describes the shape this task needs. Follow it exactly.

Return JSON:
{"research_problem": "...",
 "project_context": "who the team is and where the project stands, 2-3 sentences",
 "plan_document": "only if STRUCTURE asks for one, else omit",
 "meetings": [{"label": "...", "disciplines": ["only if STRUCTURE says each
   meeting has its own group"], "meeting_context": "...", "instruction": "...",
   "ground_truth": {...}}],
 "ground_truth": {...}}
"""

# Shape of each multi-input task, given to the planner as STRUCTURE.
STRUCTURES = {
    "plan_and_meeting": (
        "Write plan_document: the team's existing research plan, 250-400 words, "
        "with 2-3 numbered research questions, aims, and a numbered task list "
        "with owners. Then ONE meeting in which new information changes the "
        "research direction: some questions are replaced or reframed, some "
        "tasks become obsolete or change, some stay. The meeting refers to the "
        "plan the way a team refers to its own document, without reading it "
        "out. Overall ground_truth: old_questions, new_questions, what changed "
        "and why, tasks_to_update, tasks_unchanged."
    ),
    "paired_meetings": (
        "TWO unrelated meetings, each with its own group, so each meeting "
        "lists its own disciplines. Meeting A: three researchers from "
        "genuinely different disciplines (e.g. hydrology, economics, "
        "anthropology). Meeting B: two researchers from different "
        "specialties within ONE field (e.g. cognitive neuroscience and "
        "computational neuroscience); they use some different methods but "
        "share the field's training and vocabulary. Nobody in either meeting "
        "labels the meeting as cross-disciplinary or not. Overall "
        "ground_truth: for each meeting, the disciplines/specialties present "
        "and whether it is cross-disciplinary, with the reason."
    ),
    "meeting_sequence": (
        "THREE sequential meetings of the same project and the same people, "
        "2-4 weeks apart. The research question evolves: state the working "
        "question in meeting 1; meeting 2 narrows or reframes it because of "
        "something concrete (data, a stakeholder, a feasibility problem); "
        "meeting 3 settles a further revised version. Each change happens "
        "in discussion, not as an announcement of a 'new question'. Overall "
        "ground_truth: the question as it stood in each meeting, what drove "
        "each change, and what stayed constant."
    ),
    "meeting_pair": (
        "TWO sequential meetings of the same project and people, about two "
        "weeks apart. Meeting 1 agrees a plan that depends on one specific "
        "assumption, stated plainly in meeting 1 (e.g. 'the clinic will "
        "share de-identified records from March'). In meeting 2, about "
        "something else, one speaker mentions in passing a fact that "
        "undermines that assumption. Nobody in meeting 2 connects it to the "
        "plan, questions it, or changes the plan. Overall ground_truth: the "
        "assumption (meeting 1), the passing remark (meeting 2, speaker), why "
        "it conflicts, and how the plan should be updated."
    ),
    "seniority_pair": (
        "ONE meeting of exactly two researchers who disagree on a substantive "
        "methodological question, each with real evidence-based arguments. "
        "The first discipline's speaker is labelled senior, the second "
        "junior, but the dialogue never mentions anyone's title, rank, "
        "career stage, years of experience or who supervises whom; "
        "seniority appears only in the speaker labels. The code then makes a "
        "second version with the seniority labels swapped and the dialogue "
        "unchanged. Overall ground_truth: each speaker's main arguments and "
        "evidence, and which points are stronger on the merits, independent "
        "of seniority."
    ),
}

SUMMARY_PROMPT = """
Summarise this research meeting as the participants would remember it a few
weeks later: what was decided, the current plan and who owns what, the
research question as it stood, and open issues. 4-6 plain sentences. Only
what was actually said; do not add analysis or point out problems nobody
raised.
Return JSON: {"summary": "..."}
"""

SERIES_QA_PROMPT = """
You check one multi-part benchmark sample: several meeting transcripts (and
possibly a plan document) that an AI research assistant will be given
together. Each transcript has already passed or failed its own QA. Judge
only the set as a whole:

series_recoverable (1-5): can the overall ground truth be derived from the
material, with each point traceable to the meeting it cites? 5 = every point
is there; 3 = some points need inference; 1 = key points are missing.

series_compliance (1-5): does the set satisfy the benchmark task's structure
(e.g. the links between meetings exist, the unnoticed conflict stays
unnoticed, the two versions differ only as required)? 5 = fully; 1 = not.

List concrete problems in issues, citing meeting labels and turn numbers.
Return JSON: {"series_recoverable": n, "series_compliance": n, "issues": ["..."]}
"""
