"""
multi_meeting.py
Generates the five benchmark tasks that need more than one input:

    problem_02      plan_and_meeting   a research plan + a meeting that changes direction
    integration_02  paired_meetings    a cross-disciplinary meeting + a within-field one
    self_review_01  meeting_sequence   3 meetings in which the research question evolves
    adaptation_01   meeting_pair       2 meetings; a remark in the 2nd undermines the 1st's plan
    other_02        seniority_pair     1 meeting in two versions with seniority labels swapped

How it works:
1. A series planner (one call) designs the set: shared project context,
   each meeting's own instruction and ground truth, and the reference
   answer for the whole set.
2. Each meeting is filmed and QA'd by the normal Agents 2-4, as a
   single-meeting sample whose task is that meeting's part. The same cast
   is reused across meetings of one project.
3. After each meeting, a short summary of what the team decided is carried
   into the next meeting's context, so later meetings build on what was
   actually said rather than on the plan.
4. A series QA call checks the set as a whole (links between meetings,
   overall ground truth). The sample is approved only if every meeting
   passes its own QA and the set scores >= 4 on both series criteria.

Agent 1 is not used: the planner replaces it, since it has to keep several
meetings consistent with each other.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

from benchmark_tasks import get_task
from schema import (
    CastingOutput, ContentCurationOutput, DisciplineDistance, FunctionalRole, SeedSpec,
)
from utils.llm import call_llm_json, usage_report
from prompts.series_prompt import (
    SERIES_PLAN_PROMPT, STRUCTURES, SUMMARY_PROMPT, SERIES_QA_PROMPT,
)

# How many meetings the planner must return for each input type.
MEETING_COUNTS = {
    "plan_and_meeting": 1,
    "paired_meetings": 2,
    "meeting_sequence": 3,
    "meeting_pair": 2,
    "seniority_pair": 1,
}
SPEAKERS_PER_GROUP = {"seniority_pair": 2}
SERIES_THRESHOLD = 4
SENIOR, JUNIOR = FunctionalRole.PI, FunctionalRole.PHD_STUDENT


def plan_series(task, research_problem, disciplines, retries=1):
    """Plan the set; a malformed plan is re-requested before anything is filmed."""
    for attempt in range(retries + 1):
        try:
            return _plan_once(task, research_problem, disciplines)
        except ValueError as exc:
            if attempt == retries:
                raise
            print(f"      (series plan rejected, asking again: {exc})")


def meeting_group(meeting):
    """Disciplines (or specialties) of one paired meeting, or None if unusable."""
    raw = meeting.get("disciplines") or meeting.get("specialties") or []
    group = [d.strip() for d in raw if isinstance(d, str) and d.strip()]
    return group if 2 <= len(group) <= 4 and len(set(group)) == len(group) else None


def _plan_once(task, research_problem, disciplines):
    raw = call_llm_json(
        SERIES_PLAN_PROMPT,
        json.dumps({
            "benchmark_task": task["benchmark_task"],
            "STRUCTURE": STRUCTURES[task["input_type"]],
            "suggested_research_problem": research_problem,
            "disciplines": disciplines,
        }, indent=2),
        max_tokens=6000,
    )
    meetings = raw.get("meetings")
    expected = MEETING_COUNTS[task["input_type"]]
    if not isinstance(meetings, list) or len(meetings) != expected:
        raise ValueError(
            f"Series planner returned {len(meetings) if isinstance(meetings, list) else 'no'} "
            f"meetings; {task['input_type']} needs {expected}."
        )
    for m in meetings:
        for key in ("label", "meeting_context", "instruction"):
            if not isinstance(m.get(key), str) or not m[key].strip():
                raise ValueError(f"Series planner: a meeting has no {key}.")
        if not isinstance(m.get("ground_truth"), dict):
            m["ground_truth"] = {"must_contain": m["instruction"]}
    if task["input_type"] == "paired_meetings":
        # Checked here, before filming: a within-field meeting once came back
        # as ["neuroscience"] and the run crashed after meeting A was paid for.
        for m in meetings:
            if not meeting_group(m):
                raise ValueError(f"{m['label']}: needs 2-4 distinct disciplines or "
                                 f"specialties, got {m.get('disciplines')}")
    if task["input_type"] == "plan_and_meeting" and not str(raw.get("plan_document", "")).strip():
        raise ValueError("Series planner returned no plan_document.")
    if not isinstance(raw.get("ground_truth"), dict):
        raise ValueError("Series planner returned no overall ground_truth.")
    return raw


def numbered(transcript):
    return "\n".join(f"[{i}] {t.speaker}: {t.text}" for i, t in enumerate(transcript, 1))


def summarize(meeting_label, transcript):
    raw = call_llm_json(SUMMARY_PROMPT, f"{meeting_label}\n\n{numbered(transcript)}", max_tokens=600)
    return str(raw.get("summary", "")).strip()


def meeting_content(task, plan, meeting, index, total, disciplines, distance, earlier):
    """One meeting as a single-meeting sample for Agents 2-4."""
    from agents.agent1_content_curation import strip_dialogue

    context = [plan.get("project_context", ""), f"{meeting['label']}. {meeting['meeting_context']}"]
    if plan.get("plan_document") and task["input_type"] == "plan_and_meeting":
        context.append("The team's current research plan, which everyone has read:\n"
                       + plan["plan_document"])
    if earlier:
        context.append("What the team remembers from earlier meetings:\n" + "\n".join(earlier))

    # QA judges task_compliance against this, so it names only this meeting's part.
    part = (f"Part {index} of {total} of a multi-part sample for this benchmark task: "
            f"{task['benchmark_task']}\nJudge this transcript only on its own part: "
            f"{meeting['instruction']}")
    if task["input_type"] == "seniority_pair":
        part += ("\nSeniority labels are attached to the speakers afterwards, in two "
                 "swapped versions; the dialogue itself should not show seniority. "
                 "Do not mark it down for the labels being absent from the turns.")
    return ContentCurationOutput(
        research_problem=plan.get("research_problem") or "",
        disciplines=disciplines,
        discipline_distance=distance,
        meeting_context="\n\n".join(c for c in context if c),
        seed=SeedSpec(
            task_id=task["task_id"],
            benchmark_category=task["benchmark_category"],
            benchmark_task=part,
            explicit_instruction=meeting["instruction"],
            ground_truth=strip_dialogue(meeting["ground_truth"]),
        ),
    )


def series_qa(task, plan, meetings_out, versions):
    parts = [f"BENCHMARK TASK: {task['benchmark_task']}",
             f"OVERALL GROUND TRUTH:\n{json.dumps(plan['ground_truth'], indent=2)}"]
    if plan.get("plan_document"):
        parts.append(f"PLAN DOCUMENT:\n{plan['plan_document']}")
    for m in meetings_out:
        roles = ", ".join(f"{s['name']} ({s['discipline']}, {s['functional_role']})" for s in m["speakers"])
        parts.append(f"=== {m['label']} ===\nSpeakers: {roles}\n{numbered(m['_turns'])}")
    if versions:
        parts.append("VERSIONS (same dialogue, these speaker labels):\n" + json.dumps(versions, indent=2))
    raw = call_llm_json(SERIES_QA_PROMPT, "\n\n".join(parts), max_tokens=2000)

    def score(key):
        try:
            return max(1, min(5, int(raw.get(key))))
        except (TypeError, ValueError):
            return 1
    return {
        "series_recoverable": score("series_recoverable"),
        "series_compliance": score("series_compliance"),
        "issues": [str(i) for i in raw.get("issues", []) if i],
    }


def run_series(config):
    import agents.agent2_casting as agent2
    from main import load_ami_style, film_with_qa
    from utils.value_banks import sample_disciplines

    task = get_task(config["task_id"])
    task.pop("defaults", None)
    kind = task["input_type"]
    if kind not in MEETING_COUNTS:
        raise ValueError(f"{task['task_id']}: no multi-meeting runner for {kind}.")

    series_id = str(uuid.uuid4())[:8]
    created_at = datetime.now()
    print(f"\n{'='*50}\nSeries run: {series_id} ({task['task_id']}, {kind})\n{'='*50}\n")

    print("[1] Extracting AMI style...")
    ami_style_notes = load_ami_style(config)

    distance = DisciplineDistance(config.get("discipline_distance", "distant"))
    n_speakers = SPEAKERS_PER_GROUP.get(kind, config.get("num_speakers", 3))
    research_problem = config.get("research_problem")
    disciplines = config.get("disciplines")
    if research_problem is None:
        # Same bank the single-meeting runner uses (utils/research_banks.py).
        # With no disciplines set, take the background's own disciplines so
        # the problem and the cast match.
        from utils.research_banks import pick_research, research_text
        entry = pick_research(disciplines=disciplines, n=n_speakers,
                              distance=distance.value)
        if entry is None:
            raise ValueError(f"No research background covers {disciplines}. "
                             "Add one to utils/research_banks.py or set research_problem.")
        disciplines = disciplines or list(entry["disciplines"])
        research_problem = research_text(entry, only=disciplines)
    elif not disciplines:
        disciplines = sample_disciplines(distance.value, n=n_speakers)

    print("[2] Planning the series...")
    plan = plan_series(task, research_problem, disciplines)
    total = len(plan["meetings"])
    for m in plan["meetings"]:
        print(f"      {m['label']}: {m['instruction'][:90]}...")

    casts = {}  # one cast per group of disciplines, reused across meetings
    meetings_out, earlier = [], []
    for index, meeting in enumerate(plan["meetings"], 1):
        group = disciplines
        if kind == "paired_meetings":
            group = meeting_group(meeting)
        content = meeting_content(task, plan, meeting, index, total, group,
                                  distance, earlier if kind != "paired_meetings" else [])

        key = tuple(group)
        if key not in casts:
            print(f"[3] Casting for {', '.join(group)}...")
            overrides = ({group[0]: {"functional_role": SENIOR}, group[1]: {"functional_role": JUNIOR}}
                         if kind == "seniority_pair" else config.get("speaker_overrides", {}))
            casts[key] = agent2.run(content=content, speaker_overrides=overrides)
            print(f"      Cast: {', '.join(s.name for s in casts[key].speakers)}")
        casting = casts[key]

        print(f"\n--- {meeting['label']} ({index}/{total}) ---")
        filming, qa, history = film_with_qa(
            content, casting, ami_style_notes, config["max_refilms"], step=("   [film]", "   [qa]"),
        )
        meetings_out.append({
            "label": meeting["label"],
            "disciplines": group,
            "speakers": [s.model_dump(mode="json") for s in casting.speakers],
            "meeting_context": content.meeting_context,
            "seed_spec": content.seed.model_dump(),
            "ground_truth": content.seed.ground_truth,
            "transcript": [t.model_dump() for t in filming.transcript],
            "seed_location": filming.seed_location,
            "qa_feedback": qa.model_dump(),
            "qa_history": [h.model_dump() for h in history],
            "_turns": filming.transcript,
        })
        if index < total and kind != "paired_meetings":
            earlier.append(f"{meeting['label']}: {summarize(meeting['label'], filming.transcript)}")

    # other_02: same dialogue, seniority labels swapped between the two speakers.
    versions = None
    if kind == "seniority_pair":
        a, b = meetings_out[0]["speakers"][:2]
        versions = [
            {"version": "A", "labels": {a["name"]: SENIOR.value, b["name"]: JUNIOR.value}},
            {"version": "B", "labels": {a["name"]: JUNIOR.value, b["name"]: SENIOR.value}},
        ]

    print("\n[4] Series QA...")
    sqa = series_qa(task, plan, meetings_out, versions)
    meetings_ok = all(m["qa_feedback"]["approved"] for m in meetings_out)
    approved = (meetings_ok and sqa["series_recoverable"] >= SERIES_THRESHOLD
                and sqa["series_compliance"] >= SERIES_THRESHOLD)
    print(f"      Meetings approved: {sum(m['qa_feedback']['approved'] for m in meetings_out)}/{total}")
    print(f"      Series: recoverable {sqa['series_recoverable']}, compliance {sqa['series_compliance']}")
    print(f"      Approved: {approved}")

    for m in meetings_out:
        m.pop("_turns")
    out = {
        "series_id": series_id,
        "generated_at": created_at.isoformat(),
        "task_id": task["task_id"],
        "benchmark_category": task["benchmark_category"],
        "benchmark_task": task["benchmark_task"],
        "input_type": kind,
        "research_problem": plan.get("research_problem"),
        "project_context": plan.get("project_context"),
        "plan_document": plan.get("plan_document") if kind == "plan_and_meeting" else None,
        "ground_truth": plan["ground_truth"],
        "meetings": meetings_out,
        "versions": versions,
        "series_qa": sqa,
        "approved": approved,
        "ami_style_notes": ami_style_notes,
    }
    status = "approved" if approved else "failed"
    out_path = (Path(__file__).parent / "data" / "outputs" / task["task_id"] / status /
                f"{task['task_id']}_{created_at.strftime('%Y-%m-%d_%H-%M-%S')}_{series_id}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n✓ Saved to {out_path}")
    print(f"\nAPI usage this run:\n{usage_report()}")
    return out
