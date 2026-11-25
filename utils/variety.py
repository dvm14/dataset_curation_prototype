"""
variety.py
Batch-level variety metrics for generated meetings (no API calls).

The critic judges one transcript at a time; these look at the whole batch:
- setup spread: meeting style, opening, research background, disciplines,
  speaker count, overall and per task (one task reusing a background across
  its samples is what Hadley's PR #6 fixed by hand)
- shape spread: turns, words per turn, airtime balance, question rate
- repetition across transcripts: phrases, openings and closings many
  meetings share, near-duplicate meetings within a task, recurring cast names
- pass rate by setup: does a style or background fail a task systematically
  (Rita found screen_share_working_session at 0/5 by hand)
- pairing: sample k of a task should get the same setup for every generator

Every function takes records from to_record(), so they're tested offline.
"""
from __future__ import annotations

import math
import re
import statistics
from collections import Counter, defaultdict
from itertools import combinations

SHINGLE = 5                  # words per phrase for repetition and near-duplicates
NEAR_DUPLICATE = 0.25        # Jaccard of shingle sets
SUBSTANTIVE_WORDS = 12       # shorter turns are reactions and goodbyes
_STOP = set("""a an and are as at be but by for from has have i i'm i'd i'll if in is it
it's its just like me my of on or so that that's the their them then there they this to
um uh was we we're we'll what when which who will with would you your yeah okay ok""".split())


# ── records ──────────────────────────────────

def _turns(meeting):
    return [(t["speaker"], t["text"]) for t in meeting.get("transcript", [])]


def to_record(data, status=None):
    """One saved output (single meeting or multi-meeting set) as a flat record."""
    if "transcript" in data:
        meetings = [_turns(data)]
        speakers = data.get("speakers", [])
        disciplines = data.get("disciplines", [])
    else:
        meetings = [_turns(m) for m in data.get("meetings", [])]
        speakers = [s for m in data.get("meetings", []) for s in m.get("speakers", [])]
        disciplines = sorted({d for m in data.get("meetings", []) for d in m.get("disciplines", [])})
    names = sorted({s["name"] for s in speakers if s.get("name")})
    return {
        "task_id": data.get("task_id"),
        "kind": "single" if "transcript" in data else data.get("input_type", "multi"),
        "status": status or ("approved" if data.get("approved") or
                             (data.get("qa_feedback") or {}).get("approved") else "failed"),
        "generator": data.get("generator_model") or "unknown",
        "style": data.get("meeting_style"),
        "opening": data.get("meeting_opening"),
        "research_id": data.get("research_id"),
        "disciplines": tuple(sorted(disciplines)),
        "distance": data.get("discipline_distance"),
        "n_speakers": len(data.get("speakers", [])) if "transcript" in data else None,
        "seed": data.get("sample_seed"),
        "names": names,
        "meetings": meetings,
        "critic": data.get("_critic_verdict"),
    }


# ── setup spread ─────────────────────────────

def normalized_entropy(counts, universe=None):
    """1.0 = every option used equally (out of `universe` options if given)."""
    total = sum(counts.values())
    k = universe or len(counts)
    if total == 0 or k <= 1:
        return None
    h = -sum(c / total * math.log(c / total) for c in counts.values() if c)
    return round(h / math.log(k), 3)


def _key(value):
    """Tuples (discipline sets) as strings, so reports serialize to JSON."""
    return " + ".join(value) if isinstance(value, tuple) else value


def distribution(records, key, universe=None):
    counts = Counter(_key(r[key]) for r in records if r.get(key) is not None)
    out = {
        "counts": dict(counts.most_common()),
        "distinct": len(counts),
        "evenness": normalized_entropy(counts, len(universe) if universe else None),
        "top_share": round(counts.most_common(1)[0][1] / sum(counts.values()), 3) if counts else None,
    }
    if universe is not None:
        out["unused"] = sorted(set(universe) - set(counts))
    return out


def per_task_repeats(records, key, max_share=0.5, min_samples=3):
    """Tasks where one value covers more than max_share of that task's samples."""
    by_task = defaultdict(list)
    for r in records:
        if r.get(key) is not None:
            by_task[r["task_id"]].append(r[key])
    hits = []
    for task, values in sorted(by_task.items()):
        if len(values) < min_samples:
            continue
        value, n = Counter(values).most_common(1)[0]
        if n / len(values) > max_share:
            hits.append({"task_id": task, "value": value, "count": n, "of": len(values)})
    return hits


def per_task_distinct(records, key):
    """Distinct values per task, e.g. how many backgrounds each task's samples cover."""
    by_task = defaultdict(set)
    count = Counter()
    for r in records:
        if r.get(key) is not None:
            by_task[r["task_id"]].add(r[key] if not isinstance(r[key], list) else tuple(r[key]))
            count[r["task_id"]] += 1
    return {t: {"distinct": len(v), "samples": count[t]} for t, v in sorted(by_task.items())}


# ── shape ────────────────────────────────────

def shape(meeting):
    """Turns, words per turn, airtime balance and question rate for one meeting."""
    if not meeting:
        return None
    words = [len(text.split()) for _, text in meeting]
    airtime = Counter()
    for (speaker, _), n in zip(meeting, words):
        airtime[speaker] += n
    total = sum(airtime.values()) or 1
    return {
        "turns": len(meeting),
        "median_words": statistics.median(words),
        "top_airtime": round(max(airtime.values()) / total, 3),
        "question_rate": round(sum("?" in text for _, text in meeting) / len(meeting), 3),
    }


def shape_summary(records, by="generator"):
    """Median and range of each shape metric, per group (generator by default)."""
    groups = defaultdict(list)
    for r in records:
        for m in r["meetings"]:
            s = shape(m)
            if s:
                groups[r.get(by) or "unknown"].append(s)
    out = {}
    for group, shapes in sorted(groups.items()):
        out[group] = {"meetings": len(shapes)}
        for metric in ("turns", "median_words", "top_airtime", "question_rate"):
            values = [s[metric] for s in shapes]
            out[group][metric] = {"median": round(statistics.median(values), 3),
                                  "min": min(values), "max": max(values)}
    return out


# ── repetition across transcripts ────────────

def _tokens(text):
    return re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))


def shingles(meetings, n=SHINGLE):
    """Phrases of n words with at least two non-stopwords, from every turn."""
    out = set()
    for meeting in meetings:
        for _, text in meeting:
            toks = _tokens(text)
            for i in range(len(toks) - n + 1):
                gram = toks[i:i + n]
                if sum(t not in _STOP for t in gram) >= 2:
                    out.add(" ".join(gram))
    return out


def shared_phrases(records, min_share=0.1, top=25):
    """Phrases that appear in at least min_share of transcripts (document frequency)."""
    df = Counter()
    for r in records:
        df.update(shingles(r["meetings"]))
    n = len(records)
    common = [(p, c) for p, c in df.most_common() if c / n >= min_share]
    return [{"phrase": p, "transcripts": c, "share": round(c / n, 3)} for p, c in common[:top]]


def _first_words(text, k=6):
    return " ".join(_tokens(text)[:k])


def openings_and_closings(records, k=6, top=10):
    """How meetings start (first turn) and end (last substantive turn), across the batch."""
    starts, ends = Counter(), Counter()
    for r in records:
        for m in r["meetings"][:1]:
            if m:
                starts[_first_words(m[0][1], k)] += 1
        for m in r["meetings"][-1:]:
            substantive = [t for _, t in m if len(t.split()) >= SUBSTANTIVE_WORDS]
            if substantive:
                ends[_first_words(substantive[-1], k)] += 1
    n = len(records) or 1
    fmt = lambda c: [{"words": w, "transcripts": x, "share": round(x / n, 3)}
                     for w, x in c.most_common(top) if x > 1]
    return {"openings": fmt(starts), "closings": fmt(ends),
            "distinct_openings": len(starts), "distinct_closings": len(ends)}


def near_duplicates(records, threshold=NEAR_DUPLICATE):
    """Pairs of meetings for the same task whose phrase sets overlap heavily."""
    by_task = defaultdict(list)
    for i, r in enumerate(records):
        by_task[r["task_id"]].append(i)
    sets = {}
    pairs = []
    for task, idx in by_task.items():
        for a, b in combinations(idx, 2):
            for i in (a, b):
                if i not in sets:
                    sets[i] = shingles(records[i]["meetings"])
            union = sets[a] | sets[b]
            if not union:
                continue
            j = len(sets[a] & sets[b]) / len(union)
            if j >= threshold:
                pairs.append({"task_id": task, "a": a, "b": b, "jaccard": round(j, 3)})
    return sorted(pairs, key=lambda p: -p["jaccard"])


def cast_names(records, top=15):
    """Names and surnames the generator reuses (counted once per transcript)."""
    full, last = Counter(), Counter()
    for r in records:
        full.update(r["names"])
        last.update({re.sub(r",.*$", "", n).split()[-1] for n in r["names"] if n.split()})
    n = len(records) or 1
    fmt = lambda c: [{"name": k, "transcripts": v, "share": round(v / n, 3)}
                     for k, v in c.most_common(top) if v > 1]
    return {"full_names": fmt(full), "surnames": fmt(last),
            "distinct_full_names": len(full)}


# ── pass rate by setup ───────────────────────

def pass_rate_by(records, key, min_n=3):
    """Approval rate for each value of key, with the sample count."""
    groups = defaultdict(list)
    for r in records:
        value = r.get(key)
        if isinstance(value, (list, tuple)):
            value = " + ".join(value)
        if value is not None:
            groups[value].append(r["status"] == "approved")
    rows = [{"value": v, "n": len(x), "approved": sum(x), "rate": round(sum(x) / len(x), 3)}
            for v, x in groups.items() if len(x) >= min_n]
    return sorted(rows, key=lambda row: (row["rate"], -row["n"]))


def zero_pass_cells(records, key, min_n=2):
    """(task, value) cells with at least min_n samples and none approved."""
    cells = defaultdict(list)
    for r in records:
        if r.get(key) is not None:
            cells[(r["task_id"], r[key])].append(r["status"] == "approved")
    return [{"task_id": t, "value": v, "n": len(x)}
            for (t, v), x in sorted(cells.items()) if len(x) >= min_n and not any(x)]


# ── pairing across generators ────────────────

PAIRED_FIELDS = ("style", "opening", "research_id")


def pairing(records):
    """Sample k of a task should have the same setup for every generator."""
    groups = defaultdict(list)
    for r in records:
        if r.get("seed") is not None:
            groups[(r["task_id"], r["seed"])].append(r)
    checked, mismatches = 0, []
    for (task, seed), rs in sorted(groups.items()):
        if len({r["generator"] for r in rs}) < 2:
            continue
        checked += 1
        for field in PAIRED_FIELDS:
            values = {r["generator"]: r.get(field) for r in rs}
            if len(set(values.values())) > 1:
                mismatches.append({"task_id": task, "seed": seed, "field": field, "values": values})
    return {"paired_samples": checked, "mismatches": mismatches}


# ── report ───────────────────────────────────

def report(records, styles=None, openings=None, research=None):
    singles = [r for r in records if r["kind"] == "single"]
    return {
        "samples": len(records),
        "single_meetings": len(singles),
        "generators": dict(Counter(r["generator"] for r in records)),
        "approved": sum(r["status"] == "approved" for r in records),
        "setup": {
            "style": distribution(records, "style", styles),
            "opening": distribution(records, "opening", openings),
            "research_id": distribution(records, "research_id", research),
            "disciplines": distribution(singles, "disciplines"),
            "n_speakers": distribution(singles, "n_speakers"),
            "distance": distribution(singles, "distance"),
        },
        "discipline_fields": dict(Counter(d for r in singles for d in r["disciplines"]).most_common()),
        "per_task": {
            "research_repeats": per_task_repeats(records, "research_id"),
            "style_repeats": per_task_repeats(records, "style"),
            "research_distinct": per_task_distinct(records, "research_id"),
        },
        "shape": shape_summary(records),
        "repetition": {
            "phrases": shared_phrases(records),
            **openings_and_closings(records),
            "near_duplicates": near_duplicates(singles),
            "names": cast_names(records),
        },
        "pass_rate": {
            "style": pass_rate_by(records, "style"),
            "research_id": pass_rate_by(records, "research_id"),
            "generator": pass_rate_by(records, "generator", min_n=1),
            "n_speakers": pass_rate_by(singles, "n_speakers"),
            "zero_pass_style_cells": zero_pass_cells(records, "style"),
        },
        "pairing": pairing(records),
    }
