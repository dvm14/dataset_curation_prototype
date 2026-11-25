"""
Run the critic (agents/agent5_critic.py) over saved single-meeting outputs.

    python critic.py                                  # data/outputs/<GEN_MODEL>/
    python critic.py --dir data/outputs/gpt-5.6-terra --tasks jargon_02 methods_03
    python critic.py --no-llm                         # code checks only, no API cost
    python critic.py --probe 2                        # + 2 blind solver runs each, contrasted

Reads <dir>/<task>/{approved,failed}/*.json (multi-meeting outputs are
skipped) and writes, into <dir>/critic_<timestamp>/:
- <transcript file name>  one CriticReport per transcript
- critic.csv              one row per transcript, QA verdict next to critic's
- summary.md              verdict counts, QA vs critic, per-task table, top flags
The saved transcripts are not changed.
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path


def load_items(root: Path, tasks=None):
    """(path, data) for single-meeting outputs under root, sorted."""
    files = sorted(p for p in root.glob("*/*/*.json")
                   if p.parent.name in ("approved", "failed")
                   and (not tasks or p.parent.parent.name in tasks))
    items, skipped = [], 0
    for path in files:
        data = json.loads(path.read_text())
        if "transcript" not in data:
            skipped += 1                          # multi-meeting set
            continue
        items.append((path, data))
    return items, skipped


def critique(data, use_llm, probe_runs=0):
    import agents.agent5_critic as critic
    return critic.run(
        task_id=data["task_id"],
        benchmark_task=data.get("benchmark_task", ""),
        ground_truth=data.get("ground_truth", {}),
        transcript=data["transcript"],
        speakers=data["speakers"],
        use_llm=use_llm,
        probe_runs=probe_runs if use_llm else 0,
    )


def row_for(path, data, report):
    qa = data.get("qa_feedback", {})
    return {
        "task_id": data["task_id"],
        "file": path.name,
        "qa_approved": path.parent.name == "approved",
        "qa_seed_recoverable": qa.get("seed_recoverable"),
        "critic_verdict": report.verdict,
        "critic_seed_recoverable": report.seed_recoverable,
        "critic_seed_implicit": report.seed_implicit,
        "unverified_quotes": report.unverified_quotes,
        "question_rate": report.checks["question_rate"]["rate"],
        "doubled_words": report.checks["doubled_words"]["count"],
        "probe_mean_score": (report.probe or {}).get("mean_score"),
        "probe_false_alarm_rate": (report.probe or {}).get("false_alarm_rate"),
        "contrast": " | ".join(sorted({c["kind"] for c in report.contrast})),
        "reasons": " | ".join(report.reasons),
    }


def reason_kind(reason):
    """Group reasons that differ only in counts, turns or quoted words."""
    kind = reason.split(":")[0].split(" (")[0]
    kind = re.sub(r'"[^"]*"', '"…"', kind)
    return re.sub(r"\d+", "N", kind)


def summary_markdown(rows, skipped, use_llm, judge):
    n = len(rows)
    verdicts = Counter(r["critic_verdict"] for r in rows)
    lines = [
        "# Critic summary",
        "",
        f"{n} single-meeting transcripts ({skipped} multi-meeting sets skipped). "
        + (f"Critic model: {judge}." if use_llm else "Code checks only (--no-llm)."),
        "",
        "| verdict | count |", "|---|---|",
        *[f"| {v} | {verdicts.get(v, 0)} |" for v in ("pass", "flag", "block")],
        "",
        "## QA vs critic",
        "",
        "| QA \\ critic | pass | flag | block |", "|---|---|---|---|",
    ]
    for qa_label, qa_value in (("approved", True), ("failed", False)):
        cells = [sum(1 for r in rows if r["qa_approved"] == qa_value and r["critic_verdict"] == v)
                 for v in ("pass", "flag", "block")]
        lines.append(f"| {qa_label} | " + " | ".join(map(str, cells)) + " |")
    lines += ["", "QA-approved items the critic blocks are the ones to read first.", "",
              "## Per task", "", "| task | n | QA approved | critic pass | flag | block |",
              "|---|---|---|---|---|---|"]
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task_id"]].append(r)
    for task, rs in sorted(by_task.items()):
        c = Counter(r["critic_verdict"] for r in rs)
        lines.append(f"| {task} | {len(rs)} | {sum(r['qa_approved'] for r in rs)} | "
                     f"{c['pass']} | {c['flag']} | {c['block']} |")
    probed = [r for r in rows if r.get("probe_mean_score") is not None]
    if probed:
        kinds = Counter(k for r in probed for k in filter(None, r["contrast"].split(" | ")))
        scores = [r["probe_mean_score"] for r in probed]
        lines += ["", "## Critic vs blind probe", "",
                  f"{len(probed)} transcripts probed; mean blind-answer score "
                  f"{sum(scores) / len(scores):.2f} / 5.", "",
                  "| disagreement | transcripts | meaning |", "|---|---|---|",
                  f"| too_faint | {kinds['too_faint']} | critic finds a point, blind solver misses it |",
                  f"| critic_disputed | {kinds['critic_disputed']} | critic calls a point missing, solver finds it |",
                  f"| ambiguous | {kinds['ambiguous']} | solver's different answer is backed by the transcript |",
                  f"| false_alarm_bait | {kinds['false_alarm_bait']} | solver reports a problem in a control |",
                  f"| none | {sum(1 for r in probed if not r['contrast'])} | the two readings agree |"]
    reason_kinds = Counter()
    for r in rows:
        kinds = {reason_kind(reason) for reason in r["reasons"].split(" | ") if reason}
        reason_kinds.update(kinds)
    lines += ["", "## Most common reasons", "", "| reason | transcripts |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in reason_kinds.most_common(15)]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=None, help="output folder; default data/outputs/<GEN_MODEL>")
    ap.add_argument("--tasks", nargs="*", default=None, help="only these task ids")
    ap.add_argument("--limit", type=int, default=None, help="at most this many transcripts")
    ap.add_argument("--no-llm", action="store_true", help="code checks only, no API calls")
    ap.add_argument("--probe", type=int, default=0, metavar="N",
                    help="also run N blind solver attempts per transcript and contrast them")
    ap.add_argument("--workers", type=int, default=1, help="transcripts critiqued in parallel")
    args = ap.parse_args()

    from utils.llm import output_root, usage_report, JUDGE_MODEL
    root = Path(args.dir) if args.dir else output_root()
    items, skipped = load_items(root, args.tasks)
    if args.limit:
        items = items[:args.limit]
    if not items:
        print(f"No single-meeting outputs under {root}.")
        return

    out_dir = root / f"critic_{datetime.now():%Y%m%d-%H%M%S}"
    out_dir.mkdir(parents=True)
    rows = []
    def one(item):
        path, data = item
        try:
            return path, data, critique(data, use_llm=not args.no_llm, probe_runs=args.probe), None
        except Exception as e:                    # keep going; note the error
            return path, data, None, e

    # Calls are network-bound, so threads are enough; the usage meter is locked.
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for done, (path, data, report, error) in enumerate(pool.map(one, items), start=1):
            print(f"[{done}/{len(items)}] {data['task_id']:16} {path.name}")
            if error:
                print(f"   error: {error}")
                continue
            (out_dir / path.name).write_text(report.model_dump_json(indent=2))
            row = row_for(path, data, report)
            rows.append(row)
            print(f"   QA {'approved' if row['qa_approved'] else 'failed  '} -> critic {report.verdict}"
                  + (f": {report.reasons[0]}" if report.reasons else ""), flush=True)

    if not rows:
        print("Nothing was critiqued.")
        return
    with open(out_dir / "critic.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "summary.md").write_text(
        summary_markdown(rows, skipped, not args.no_llm, JUDGE_MODEL))
    if not args.no_llm:
        print(usage_report())
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
