"""
Re-score saved single-meeting transcripts with another judge, without
regenerating them. Used to check whether a change in pass rate comes from
the generator or from the judge.

    python rejudge.py --judge gpt-5.6-terra
    python rejudge.py --judge gpt-5.6-terra --tasks jargon_01 problem_03

Reads data/outputs/<GEN_MODEL>/<task>/{approved,failed}/*.json (multi-meeting
outputs are skipped), runs Agent 4 with --judge, and writes a CSV that puts
the saved scores next to the new ones. The saved JSONs are not changed.
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge", required=True, help="model to re-score with, e.g. gpt-5.6-terra")
    ap.add_argument("--tasks", nargs="*", default=None, help="only these task ids")
    ap.add_argument("--limit", type=int, default=None, help="at most this many transcripts")
    args = ap.parse_args()

    os.environ["JUDGE_MODEL"] = args.judge      # read when utils.llm is imported
    from utils.llm import output_root, usage_report
    from schema import (ContentCurationOutput, CastingOutput, FilmingOutput,
                        SeedSpec, SpeakerProfile, Turn)
    import agents.agent4_qa as agent4
    keys = list(agent4.APPROVAL_THRESHOLDS)

    files = sorted(p for p in output_root().glob("*/*/*.json")
                   if p.parent.name in ("approved", "failed")
                   and (not args.tasks or p.parent.parent.name in args.tasks))
    rows = []
    for path in files:
        d = json.loads(path.read_text())
        if "transcript" not in d or "seed_spec" not in d:
            continue                                # multi-meeting output
        if args.limit and len(rows) >= args.limit:
            break
        speakers = [SpeakerProfile(**s) for s in d["speakers"]]
        content = ContentCurationOutput(
            research_problem=d["research_problem"], disciplines=d["disciplines"],
            discipline_distance=d["discipline_distance"], meeting_context="",
            seed=SeedSpec(**d["seed_spec"]))
        filming = FilmingOutput(transcript=[Turn(**t) for t in d["transcript"]],
                                seed_location=d.get("seed_location", ""))
        print(f"{d['task_id']:16} {path.name}")
        try:
            qa = agent4.run(filming=filming, content=content,
                            casting=CastingOutput(speakers=speakers),
                            ami_style_notes=d.get("ami_style_notes", ""))
        except Exception as e:                      # keep going; note the error
            print(f"   error: {e}")
            continue
        old = d["qa_feedback"]
        row = {"task_id": d["task_id"], "file": path.name,
               "old_judge": d.get("judge_model", ""), "new_judge": args.judge,
               "old_approved": old.get("approved"), "new_approved": qa.approved}
        for k in keys:
            row[f"old_{k}"] = old.get(k)
            row[f"new_{k}"] = getattr(qa, k)
        row["new_issues"] = " | ".join(qa.issues)
        rows.append(row)
        print(f"   approved {row['old_approved']} -> {row['new_approved']}   "
              + "  ".join(f"{k[:4]} {row['old_'+k]}->{row['new_'+k]}" for k in keys))

    if not rows:
        print("Nothing to re-score.")
        return
    out = output_root() / f"rejudge_{args.judge}_{datetime.now():%Y%m%d-%H%M%S}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    print(f"\n{n} transcripts   approved: {sum(r['old_approved'] for r in rows)} (saved judge) "
          f"vs {sum(r['new_approved'] for r in rows)} ({args.judge})")
    for k in keys:
        diff = sum(r[f"new_{k}"] - r[f"old_{k}"] for r in rows) / n
        print(f"  {k:22} mean change {diff:+.2f}")
    print(usage_report())
    print(f"Saved to {out}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
