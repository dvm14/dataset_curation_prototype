"""
The question the blind probe (agents/agent6_probe.py) asks for each
single-meeting task: what an evaluated assistant would be asked, written so it
doesn't give the answer away.

The task descriptions in benchmark_tasks.py can't be used as is: they say what
is planted ("an early constraint makes a later decision unworkable") and, for
controls, what the right answer is ("recognize that intervention is
unnecessary"). These questions name the job, not the finding, so a solver that
misses a faint cue gets no hint. Controls ask the same question as the task
they control, so a solver that reports a problem there is a false alarm.

These are a first draft for the team to review: they should match how the
benchmark will actually prompt assistants.
"""

PROBE_QUESTIONS = {
    "jargon_01": "Identify any terms the participants use with different, discipline-specific "
                 "meanings. For each term, give each meaning and the discipline that uses it.",
    "jargon_02": "Review this meeting for communication problems between the disciplines. "
                 "Alert the team to anything they should clarify, or say there is nothing.",
    "jargon_03": "Review this meeting for terminology problems, such as a term used in "
                 "different senses or never defined. List any you find, or say there are none.",
    "assumptions_01": "List the assumptions, stated or unstated, that the team's decisions rely "
                      "on, with the turns that show each one.",
    "assumptions_02": "Compare how the participants judge what counts as convincing evidence. "
                      "Explain each view fairly without taking a side.",
    "assumptions_03": "Review the decisions made in this meeting. Is there anything that will not "
                      "work as planned? Cite the turns, or say there is nothing.",
    "assumptions_04": "Recommend a next step for the team. You may ask one clarification question "
                      "first; if you do, state it and what you would recommend for each likely answer.",
    "problem_01": "Summarize the discussion, including the research problem and the disciplines "
                  "taking part.",
    "problem_03": "Which research question did the team choose, and which questions are still open?",
    "problem_04": "What shared research question is the team working toward, and what question "
                  "does each discipline bring to it?",
    "integration_01": "Identify the disciplines in this meeting and summarize each participant's "
                      "perspective by its substance.",
    "integration_03": "Identify the research areas in this collaboration and the role each "
                      "participant plays in it.",
    "disagreement_01": "Map the agreements and disagreements in this meeting to the disciplines "
                       "involved, with each side's reasons, evidence and any common ground.",
    "disagreement_02": "Summarize the team's decision about which data to use, including how much "
                       "support it had, and suggest when the team should revisit it.",
    "disagreement_03": "Classify the substantive exchanges in this meeting as agreement, "
                       "disagreement or neutral, with the turns for each.",
    "methods_01": "Suggest methods from each discipline that would help with this research "
                  "problem, and say why each fits.",
    "methods_02": "Summarize the methods proposed in this meeting and what problem each one is "
                  "designed to avoid.",
    "methods_03": "Assess whether the team's chosen design can answer its research question. If "
                  "it cannot, suggest a design that could.",
    "opportunity_01": "Recommend related literature that would bring a viewpoint not yet "
                      "represented in this discussion.",
    "opportunity_02": "Propose follow-up research questions that integrate the disciplines in "
                      "this meeting.",
    "opportunity_03": "Is any important subtopic missing from this discussion? Explain, or say "
                      "nothing is missing.",
    "opportunity_04": "Which perspectives or fields, if any, are missing from this team for its "
                      "research problem? Say none if none are needed.",
    "coordination_01": "List the next tasks from this meeting and which participant or "
                       "discipline takes each one.",
    "coordination_02": "Suggest appropriate next steps for the team.",
    "coordination_03": "Lay out the project timeline from this meeting and flag any scheduling "
                       "problems, or say there are none.",
    "coordination_04": "Put the next steps in the order they can be done and explain any "
                       "dependencies between them.",
    "coordination_05": "List the action items from this meeting with their owners and dates, "
                       "exactly as they were stated.",
    "other_01": "Does this team need an intervention from you to work well together? If so, "
                "what would you do?",
}

# Tasks where the right answer is "no problem here": the probe counts any
# problem the solver reports as a false alarm.
CONTROL_TASKS = {"jargon_03", "opportunity_04", "other_01"}


def probe_question(task_id: str) -> str:
    if task_id not in PROBE_QUESTIONS:
        raise ValueError(f"No probe question for {task_id}; add one to probe_questions.py")
    return PROBE_QUESTIONS[task_id]
