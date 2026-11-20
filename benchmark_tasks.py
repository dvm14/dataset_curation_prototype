"""33 tasks from Evaluation Benchmark v2 (2026-10-31).
Descriptions are faithful implementation summaries, not verbatim quotations.
No generation_requirements field: benchmark_task carries the scenario conditions.
input_type is routing metadata, not an additional generation prompt.
"""
from copy import deepcopy

_ROWS = [
    ('jargon_01', 'Jargon', 'single_meeting', 'Give a meeting transcript where two researchers from different disciplines use the same term with different discipline-specific definitions. The evaluated AI must identify both meanings and associate each with the correct discipline.'),
    ('jargon_02', 'Jargon', 'single_meeting', 'Give a meeting where researchers from two disciplines use the same term in different contexts without realizing the mismatch. Leave the misunderstanding unnoticed. The evaluated AI must alert the team to possible terminology confusion.'),
    ('jargon_03', 'Jargon', 'single_meeting', 'CONTROL: Give a meeting where every technical term is defined early and used consistently. The evaluated AI is asked to flag terminology problems; false alarms are errors. Do not plant terminology confusion.'),
    ('assumptions_01', 'Assumptions', 'single_meeting', 'Give a meeting where researchers finalize decisions relying on both explicit and implicit assumptions. The evaluated AI must identify the decision-relevant assumptions. Provide a proposed reference answer with evidence; implicit assumptions should be inferable, not invented.'),
    ('assumptions_02', 'Assumptions', 'single_meeting', 'Give a meeting between distant disciplines with different standards of evidence, neither explicitly naming that difference; for example trial data versus interviews. The evaluated AI must explain the differing standards fairly without choosing a side.'),
    ('assumptions_03', 'Assumptions', 'single_meeting', 'Give a long meeting where an early constraint makes a later decision unworkable, and nobody connects them. Separate the two moments with substantive discussion. The evaluated AI must flag the conflict and cite both moments.'),
    ('assumptions_04', 'Assumptions', 'single_meeting', 'Give a conversation where a consequential assumption remains ambiguous between plausible interpretations. The evaluated AI may ask one clarification question before recommending a next step. Preserve the ambiguity and record interpretations and information needed to resolve it.'),
    ('problem_01', 'Problem Formulation', 'single_meeting', 'Give an interdisciplinary research discussion with an identifiable research problem. The evaluated AI must summarize the whole discussion and acknowledge the participating disciplines.'),
    ('problem_02', 'Problem Formulation', 'plan_and_meeting', 'Provide an existing research plan and a new meeting that changes the research direction. The evaluated AI must identify old and new questions, explain changes, update relevant tasks and preserve the earlier direction.'),
    ('problem_03', 'Problem Formulation', 'single_meeting', 'Give a meeting that considers multiple research questions and ultimately chooses one. Preserve which alternatives remain unresolved. The evaluated AI must accurately track those unresolved questions.'),
    ('problem_04', 'Problem Formulation', 'single_meeting', 'Give a meeting with three disciplines, each opening with a different research question, that loosely converge on a shared question. The evaluated AI must capture the bridging question while preserving all three disciplinary sub-questions.'),
    ('integration_01', 'Integration/Perspectives', 'single_meeting', 'Give a meeting where one discipline dominates airtime and a quieter participant raises a substantive point that should change the plan. The evaluated AI must identify disciplines and summarize perspectives by substance, preserving the quieter contribution.'),
    ('integration_02', 'Integration/Perspectives', 'paired_meetings', 'Provide a pair of transcripts: a genuinely cross-disciplinary meeting and a meeting between two specialties within the same field. The evaluated AI must accurately identify disciplines without treating every specialty as a separate discipline.'),
    ('integration_03', 'Integration/Perspectives', 'single_meeting', 'Give a meeting with three fields where one participant provides a method or tool, without an independent research question. The evaluated AI must distinguish this supporting role from a third core research area with its own goals.'),
    ('disagreement_01', 'Disagreement/Agreement', 'single_meeting', 'Give a meeting containing agreements and disagreements across disciplines. The evaluated AI must map each to the correct disciplines and explain each side\'s rationale, evidence and common ground.'),
    ('disagreement_02', 'Disagreement/Agreement', 'single_meeting', 'Give a meeting where the team chooses a dataset, one member objects once with a reason, and then stays silent about that decision. The evaluated AI must preserve the dissent and reason, propose a sensible revisit trigger, and not call the decision unanimous.'),
    ('disagreement_03', 'Disagreement/Agreement', 'single_meeting', 'Give a meeting with two indirect disagreements, softened into questions or comments about disciplinary practice rather than explicit statements of disagreement. The evaluated AI must recognize them instead of labeling them agreement or neutral comments.'),
    ('methods_01', 'Methods', 'single_meeting', 'Give an interdisciplinary meeting with enough research context for the evaluated AI to identify helpful methods from each discipline. Proposed reference methods must fit both the discipline and the research problem; expert review is needed.'),
    ('methods_02', 'Methods', 'single_meeting', 'Give a meeting where two disciplines propose conflicting methods for the same step, with credible reasons for both. The evaluated AI must fairly summarize both and explain what problem each method avoids, rather than simply selecting a winner.'),
    ('methods_03', 'Methods', 'single_meeting', 'Give a meeting where the team chooses a method incapable of answering its research question, for example a one-time survey for a causal question. Leave the mismatch for the evaluated AI to detect and suggest a suitable design.'),
    ('opportunity_01', 'Opportunity Generation', 'single_meeting', 'Give an interdisciplinary research meeting with sufficient topic context for the evaluated AI to recommend related literature offering a new viewpoint. Record existing perspectives so novelty can be assessed. Do not fabricate publications; recommended literature needs external verification.'),
    ('opportunity_02', 'Opportunity Generation', 'single_meeting', 'Give an interdisciplinary meeting with a main research question. The evaluated AI must propose relevant follow-up questions integrating the disciplines; novelty, usefulness and feasibility require expert judgment.'),
    ('opportunity_03', 'Opportunity Generation', 'single_meeting', 'Give a meeting that omits substantive discussion or rationale on an important subtopic. Supply enough contextual evidence to infer the gap without having participants announce the missing topic. The evaluated AI must identify that gap.'),
    ('opportunity_04', 'Opportunity Generation', 'single_meeting', 'CONTROL: Give a meeting where all perspectives needed for the research problem are already represented. The evaluated AI is asked which perspectives are missing; unnecessary additional fields count as errors. Do not plant a missing perspective.'),
    ('coordination_01', 'Coordination', 'single_meeting', 'Give a meeting planning research next steps with 2-4 disciplines represented. The evaluated AI must map tasks to disciplines consistently with the discussion.'),
    ('coordination_02', 'Coordination', 'single_meeting', 'Give a meeting with a clear research direction but little explicit next-step planning. The evaluated AI must suggest appropriate next steps. Preserve the lack of an explicit complete action plan.'),
    ('coordination_03', 'Coordination', 'single_meeting', 'Give a meeting where approval is expected on March 1, 2027 but interviews requiring that approval are planned for February 1, 2027. Leave the timeline conflict unresolved for the evaluated AI to flag, including the affected steps.'),
    ('coordination_04', 'Coordination', 'single_meeting', 'Give a meeting discussing steps out of order, where work by one discipline must finish before another discipline can start. The evaluated AI must order next steps by dependency and name the dependency issue.'),
    ('coordination_05', 'Coordination', 'single_meeting', 'Give a meeting ending with exactly six action items: some with explicit owners and dates, others without owners. The evaluated AI must preserve assigned owners and dates and mark the remaining items unassigned instead of inventing owners.'),
    ('self_review_01', 'Self-review', 'meeting_sequence', 'Provide 3-5 sequential meetings for one project in which the research question evolves. The evaluated AI must accurately map the evolution over time.'),
    ('adaptation_01', 'Adaptation across meetings', 'meeting_pair', 'Provide two sequential meetings. A passing remark in the second undermines an assumption behind the first meeting\'s plan, but nobody connects them. The evaluated AI must link the new fact to the earlier assumption and flag the conflict while updating the plan.'),
    ('other_01', 'Other', 'single_meeting', 'CONTROL: Give a productive exchange where researchers use different approaches but understand their respective roles and limitations. The evaluated AI must recognize that intervention is unnecessary, without inventing disagreement or forcing integration.'),
    ('other_02', 'Other', 'seniority_pair', 'Provide two versions of the same discussion with identical arguments but swapped speaker-seniority labels. The evaluated AI\'s assessment should remain substantively consistent and evidence-based. Do not independently regenerate different arguments.'),
]
# Per-task run defaults, used only when CONFIG leaves the setting unset
# (e.g. disciplines=None). Kept here rather than as special cases in main.py.
_DEFAULTS = {
    # The task needs two distant disciplines with different evidence
    # standards; the description's own example is trial data vs interviews.
    'assumptions_02': dict(disciplines=['clinical medicine', 'sociology']),
}

TASKS = {key: dict(task_id=key, benchmark_category=category,
                  benchmark_task=description, input_type=kind,
                  defaults=_DEFAULTS.get(key, {}))
         for key, category, kind, description in _ROWS}

def get_task(task_id):
    if task_id not in TASKS:
        raise ValueError(f'Unknown task_id: {task_id}. Use --list-tasks.')
    return deepcopy(TASKS[task_id])

def require_single_meeting(task):
    if task['input_type'] != 'single_meeting':
        raise ValueError(f"{task['task_id']} requires {task['input_type']}. "
                         "This patch only generates single-meeting samples; "
                         "a multi-input orchestrator is still required. No API calls made.")
