"""
run_batch.py
Generate N samples for every task (or a chosen subset), several at a time.

    python run_batch.py --n 1                    # pilot: one sample per task
    python run_batch.py --n 10 --workers 6       # full run
    python run_batch.py --n 10 --tasks jargon_01 coordination_03
    python run_batch.py --n 10 --budget 150      # stop starting new runs past $150
    python run_batch.py --n 10 --dry-run         # show what would run, spend nothing

Models come from .env (GEN_MODEL, JUDGE_MODEL; see utils/llm.py). Samples are
saved under data/outputs/<GEN_MODEL>/<task_id>/{approved,failed}/, one JSON per
sample (a multi-meeting sample holds all its meetings).

Resumable: samples already on disk count toward N, approved or failed, so an
interrupted run picks up where it stopped. Each sample runs as its own
`python main.py --task ...` process with its log in <output root>/logs/.
A run that crashes is retried once. The cost guard uses each run's own cost
report, so it can overshoot by up to the runs already in flight.
"""

import argparse
import csv
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from benchmark_tasks import TASKS
from utils.llm import GEN_MODEL, JUDGE_MODEL, output_root

ROOT = Path(__file__).resolve().parent
COST_RE = re.compile(r"total ≈ \$([0-9.]+)")


def existing(task_id):
    folder = output_root() / task_id
    return sum(len(list((folder / s).glob("*.json"))) for s in ("approved", "failed"))


def run_one(task_id, index, log_dir):
    """Run one sample. Returns (status, cost, seconds, log path)."""
    log = log_dir / f"{task_id}_{index:02d}_{datetime.now():%Y%m%d-%H%M%S}.log"
    start = time.time()
    with open(log, "w") as fh:
        code = subprocess.call(
            [sys.executable, "-u", "main.py", "--task", task_id, "--seed", str(index)],
            cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
        )
    text = log.read_text(errors="replace")
    match = COST_RE.findall(text)
    cost = float(match[-1]) if match else 0.0
    status = {0: "approved", 2: "failed"}.get(code, "crashed")
    return status, cost, time.time() - start, log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, required=True, help="samples wanted per task, counting ones already saved")
    ap.add_argument("--tasks", nargs="*", default=list(TASKS), help="task ids (default: all 33)")
    ap.add_argument("--workers", type=int, default=4, help="samples generated at the same time")
    ap.add_argument("--budget", type=float, default=None, help="stop starting new runs once this many $ are spent")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    unknown = [t for t in args.tasks if t not in TASKS]
    if unknown:
        ap.error(f"unknown task ids: {', '.join(unknown)}")

    jobs = []
    for task_id in args.tasks:
        have = existing(task_id)
        jobs += [(task_id, i) for i in range(have + 1, args.n + 1)]

    print(f"Generator: {GEN_MODEL}   Judge: {JUDGE_MODEL}")
    print(f"Output:    {output_root()}")
    print(f"Tasks: {len(args.tasks)}   wanted per task: {args.n}   to run now: {len(jobs)}"
          f"   workers: {args.workers}" + (f"   budget: ${args.budget:.2f}" if args.budget else ""))
    if args.dry_run or not jobs:
        for task_id in args.tasks:
            print(f"  {task_id:16} have {existing(task_id):3}  run {sum(j[0] == task_id for j in jobs):3}")
        return

    log_dir = output_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    spent = 0.0
    stop = threading.Event()
    results = defaultdict(lambda: {"approved": 0, "failed": 0, "crashed": 0, "cost": 0.0})
    done = 0

    def work(task_id, index):
        nonlocal spent, done
        if stop.is_set():
            return
        for attempt in (1, 2):
            status, cost, secs, log = run_one(task_id, index, log_dir)
            with lock:
                spent += cost
                results[task_id]["cost"] += cost
                if status != "crashed" or attempt == 2:
                    results[task_id][status] += 1
                    done += 1
                print(f"[{done:3}/{len(jobs)}] {task_id:16} #{index:<2} {status:8} ${cost:5.2f}  "
                      f"{secs/60:4.1f} min   total ${spent:7.2f}"
                      + ("" if status != "crashed" else f"   (log: {log.name})"), flush=True)
                if args.budget and spent >= args.budget and not stop.is_set():
                    stop.set()
                    print(f"Budget ${args.budget:.2f} reached: no new runs will start.", flush=True)
            if status != "crashed":
                return

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(work, *job) for job in jobs]
            for f in as_completed(futures):
                f.result()
    except KeyboardInterrupt:
        stop.set()
        print("\nInterrupted: finishing the runs in progress, starting no new ones...")

    # summary
    print(f"\n{'task':16} {'approved':>8} {'failed':>7} {'crashed':>8} {'on disk':>8} {'cost':>8}")
    rows = []
    for task_id in args.tasks:
        r = results[task_id]
        rows.append([task_id, r["approved"], r["failed"], r["crashed"], existing(task_id), round(r["cost"], 2)])
        print(f"{task_id:16} {r['approved']:8} {r['failed']:7} {r['crashed']:8} {existing(task_id):8} ${r['cost']:7.2f}")
    print(f"{'TOTAL':16} {sum(r[1] for r in rows):8} {sum(r[2] for r in rows):7} "
          f"{sum(r[3] for r in rows):8} {sum(r[4] for r in rows):8} ${spent:7.2f}")

    summary = output_root() / f"batch_summary_{datetime.now():%Y%m%d-%H%M%S}.csv"
    with open(summary, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["task_id", "approved", "failed", "crashed", "on_disk", "cost_usd"])
        w.writerows(rows)
    print(f"\nSummary saved to {summary}")


if __name__ == "__main__":
    main()
