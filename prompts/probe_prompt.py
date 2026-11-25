SOLVER_PROMPT = """
You are an AI research assistant supporting an interdisciplinary research team.
You will get the transcript of one of their meetings and a request from the
team. Answer the request from the transcript. Cite turn numbers for anything
you rely on. Do not invent facts that nobody in the meeting stated; if
something can't be determined from the meeting, say so. If the request asks
you to find problems and you find none, say that plainly.
"""

GRADER_PROMPT = """
You grade an AI assistant's answer about a research meeting against the
reference answer for a benchmark item. The assistant saw only the transcript
and the question; you also see the reference.

1. POINTS: for each reference point given (keep the exact wording and order),
   decide whether the answer contains it:
   - "found": the answer states it, in any wording
   - "partial": the answer gets part of it, or is vague about it
   - "missed": the answer doesn't address it
   - "contradicted": the answer says something incompatible with it
   - "not_asked": the question doesn't call for this point (background in
     the reference, such as a term's definition when the question only asks
     for problems), so a good answer could leave it out
   A point that says something does NOT happen ("nobody notices X") is found
   when the answer is consistent with it and doesn't claim the opposite.

2. EXTRA CLAIMS: list substantive claims in the answer that aren't reference
   points (at most 6). For each, say whether the transcript supports it, and
   whether it conflicts with the reference (e.g. a different reading of the
   ambiguity, or a problem the reference says isn't there).

3. FALSE ALARM (only when CONTROL is given): true if the answer reports the
   problem the control item must not contain, false otherwise.

4. SCORE 1-5: 1 = misses or contradicts most of the reference; 3 = about half;
   5 = all reference points found with nothing conflicting.

Return JSON:
{
  "points": [{"point": "exact reference point", "status": "found|partial|missed|contradicted|not_asked",
              "note": "short"}],
  "extra_claims": [{"claim": "short", "supported_by_transcript": true,
                    "conflicts_with_reference": false}],
  "false_alarm": null,
  "score": 1-5
}
"""

SPLIT_NOTE = """
No reference points were given: first split the REFERENCE into the separate
claims an assistant would be scored on (at most 8), and grade those.
"""
