"""
Batch-level variety report for saved outputs (no API calls).

    python variety.py                                   # data/outputs/<GEN_MODEL>/
    python variety.py --dir data/outputs/gpt-5.6-terra data/outputs/claude-sonnet-5-5
    python variety.py --dir <new batch> --compare <old batch>

Reads <dir>/<task>/{approved,failed}/*.json, single meetings and multi-meeting
sets. Several --dir folders are read as one batch (e.g. GPT + Claude, which
also checks that paired samples got the same setup). With --compare, the
report adds a section on what changed between the two batches. Writes
variety.md and variety.json into the first --dir (or --out).

If a critic run (critic.py) is in the folder, its latest verdicts are added.
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def load_records(roots):
    from utils.variety import to_record
    records = []
    for root in roots:
        verdicts = latest_critic_verdicts(root)
        for path in sorted(root.glob("*/*/*.json")):
            if path.parent.name not in ("approved", "failed"):
                continue
            data = json.loads(path.read_text())
            data["_critic_verdict"] = verdicts.get(path.name)
            records.append(to_record(data, status=path.parent.name))
    return records


def latest_critic_verdicts(root):
    verdicts = {}
    for run in sorted(root.glob("critic_*")):
        for f in run.glob("*.json"):
            verdicts[f.name] = json.loads(f.read_text()).get("verdict")
    return verdicts


def banks():
    from utils.meeting_styles import MEETING_STYLES, MEETING_OPENINGS
    from utils.research_banks import RESEARCH_BANK
    return list(MEETING_STYLES), list(MEETING_OPENINGS), [e["id"] for e in RESEARCH_BANK]


def _short(value, n=60):
    s = str(value)
    return s if len(s) <= n else s[: n - 1] + "…"


def _dist_table(dist, label, top=None):
    rows = list(dist["counts"].items())[:top] if top else list(dist["counts"].items())
    lines = [f"| {label} | samples |", "|---|---|"]
    lines += [f"| {_short(' + '.join(k) if isinstance(k, tuple) else k)} | {v} |" for k, v in rows]
    return lines


def markdown(rep, title, compare=None):
    s = rep["setup"]
    L = [f"# {title}", "",
         f"{rep['samples']} samples ({rep['single_meetings']} single meetings), "
         f"{rep['approved']} approved by QA. Generators: "
         + ", ".join(f"{g} ({n})" for g, n in rep["generators"].items()) + ".", "",
         "## Setup spread", "",
         "Evenness is 1.0 when every option in the bank is used equally.", "",
         "| dimension | distinct | evenness | most common share | unused from bank |",
         "|---|---|---|---|---|"]
    for key in ("style", "opening", "research_id", "disciplines", "n_speakers", "distance"):
        d = s[key]
        unused = d.get("unused")
        L.append(f"| {key} | {d['distinct']} | {d['evenness']} | {d['top_share']} | "
                 + (", ".join(_short(u, 30) for u in unused) or "none" if unused is not None else "–")
                 + " |")
    L += ["", *_dist_table(s["style"], "meeting style"), "",
          *_dist_table(s["research_id"], "research background"), "",
          "| field | samples |", "|---|---|",
          *[f"| {k} | {v} |" for k, v in rep["discipline_fields"].items()], ""]

    pt = rep["per_task"]
    L += ["## Per task", ""]
    distinct = pt["research_distinct"]
    low = sorted(distinct.items(), key=lambda kv: kv[1]["distinct"] / kv[1]["samples"])[:8]
    L += ["Fewest distinct research backgrounds for their sample count:", "",
          "| task | backgrounds | samples |", "|---|---|---|",
          *[f"| {t} | {v['distinct']} | {v['samples']} |" for t, v in low], ""]
    for name, hits in (("research background", pt["research_repeats"]), ("style", pt["style_repeats"])):
        if hits:
            L += [f"Tasks where one {name} covers over half the samples:", "",
                  "| task | value | samples |", "|---|---|---|",
                  *[f"| {h['task_id']} | {h['value']} | {h['count']}/{h['of']} |" for h in hits], ""]
        else:
            L += [f"No task has one {name} in over half its samples.", ""]

    L += ["## Shape", "", "| generator | meetings | turns | median words/turn | top speaker's airtime | question rate |",
          "|---|---|---|---|---|---|"]
    for g, v in rep["shape"].items():
        f = lambda m: f"{v[m]['median']} ({v[m]['min']}–{v[m]['max']})"
        L.append(f"| {g} | {v['meetings']} | {f('turns')} | {f('median_words')} | "
                 f"{f('top_airtime')} | {f('question_rate')} |")

    r = rep["repetition"]
    L += ["", "## Repetition across transcripts", "",
          f"{r['distinct_openings']} distinct first-turn openings and {r['distinct_closings']} "
          f"distinct last substantive turns across {rep['samples']} samples.", ""]
    if r["openings"]:
        L += ["| meetings open with | transcripts |", "|---|---|",
              *[f"| {o['words']}… | {o['transcripts']} |" for o in r["openings"]], ""]
    if r["closings"]:
        L += ["| last substantive turn starts | transcripts |", "|---|---|",
              *[f"| {o['words']}… | {o['transcripts']} |" for o in r["closings"]], ""]
    if r["phrases"]:
        L += ["Phrases in at least 10% of transcripts:", "", "| phrase | transcripts | share |", "|---|---|---|",
              *[f"| {p['phrase']} | {p['transcripts']} | {p['share']:.0%} |" for p in r["phrases"]], ""]
    names = r["names"]
    L += [f"{names['distinct_full_names']} distinct cast names.", ""]
    if names["full_names"]:
        L += ["| reused name | transcripts |", "|---|---|",
              *[f"| {n['name']} | {n['transcripts']} |" for n in names["full_names"][:10]], ""]
    if names["surnames"]:
        L += ["| reused surname | transcripts |", "|---|---|",
              *[f"| {n['name']} | {n['transcripts']} |" for n in names["surnames"][:10]], ""]
    dupes = r["near_duplicates"]
    L += [f"Near-duplicate meeting pairs within a task (phrase overlap ≥ 0.25): {len(dupes)}.", ""]

    p = rep["pass_rate"]
    L += ["## Pass rate by setup", "", "| meeting style | samples | approved | rate |", "|---|---|---|---|",
          *[f"| {x['value']} | {x['n']} | {x['approved']} | {x['rate']:.0%} |" for x in p["style"]], "",
          "| research background (lowest 8) | samples | approved | rate |", "|---|---|---|---|",
          *[f"| {x['value']} | {x['n']} | {x['approved']} | {x['rate']:.0%} |" for x in p["research_id"][:8]], ""]
    cells = p["zero_pass_style_cells"]
    if cells:
        L += [f"Task + style cells with 2+ samples and none approved ({len(cells)}):", "",
              "| task | style | samples |", "|---|---|---|",
              *[f"| {c['task_id']} | {c['value']} | {c['n']} |" for c in cells], ""]

    pa = rep["pairing"]
    L += ["## Pairing across generators", "",
          (f"{pa['paired_samples']} task + seed samples exist for more than one generator; "
           f"{len(pa['mismatches'])} setup mismatches." if pa["paired_samples"]
           else "Only one generator in this batch, so pairing can't be checked."), ""]
    if compare:
        L += compare_markdown(rep, compare)
    return "\n".join(L) + "\n"


def compare_markdown(new, old):
    """What changed between an older batch (old) and this one (new)."""
    L = ["## Compared with the other batch", "",
         "| measure | other batch | this batch |", "|---|---|---|",
         f"| samples | {old['samples']} | {new['samples']} |",
         f"| approved | {old['approved']} ({old['approved'] / max(old['samples'], 1):.0%}) | "
         f"{new['approved']} ({new['approved'] / max(new['samples'], 1):.0%}) |"]
    for key in ("style", "opening", "research_id", "disciplines"):
        L.append(f"| {key} distinct / evenness | {old['setup'][key]['distinct']} / "
                 f"{old['setup'][key]['evenness']} | {new['setup'][key]['distinct']} / "
                 f"{new['setup'][key]['evenness']} |")
    for g in sorted(set(old["shape"]) | set(new["shape"])):
        o, n = old["shape"].get(g), new["shape"].get(g)
        for m in ("turns", "median_words", "question_rate"):
            L.append(f"| {g} {m} (median) | {o[m]['median'] if o else '–'} | {n[m]['median'] if n else '–'} |")
    L += [f"| distinct openings | {old['repetition']['distinct_openings']} | {new['repetition']['distinct_openings']} |",
          f"| phrases in ≥10% of transcripts | {len(old['repetition']['phrases'])} | {len(new['repetition']['phrases'])} |",
          f"| distinct cast names | {old['repetition']['names']['distinct_full_names']} | "
          f"{new['repetition']['names']['distinct_full_names']} |", ""]
    old_rates = {x["value"]: x for x in old["pass_rate"]["style"]}
    L += ["| meeting style | other batch | this batch |", "|---|---|---|"]
    for x in sorted(new["pass_rate"]["style"], key=lambda x: x["value"]):
        o = old_rates.get(x["value"])
        before = f"{o['rate']:.0%} of {o['n']}" if o else "–"
        L.append(f"| {x['value']} | {before} | {x['rate']:.0%} of {x['n']} |")
    return L + [""]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", nargs="*", default=None, help="batch folder(s); default data/outputs/<GEN_MODEL>")
    ap.add_argument("--compare", default=None, help="another batch folder to compare against")
    ap.add_argument("--out", default=None, help="folder for variety.md / variety.json")
    ap.add_argument("--title", default="Variety report")
    args = ap.parse_args()

    from utils.variety import report
    from utils.llm import output_root
    roots = [Path(d) for d in args.dir] if args.dir else [output_root()]
    records = load_records(roots)
    if not records:
        print(f"No outputs under {', '.join(map(str, roots))}.")
        return
    styles, openings, research = banks()
    rep = report(records, styles, openings, research)
    old = None
    if args.compare:
        old_records = load_records([Path(args.compare)])
        old = report(old_records, styles, openings, research) if old_records else None
    out = Path(args.out) if args.out else roots[0]
    out.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    (out / f"variety_{stamp}.json").write_text(json.dumps({"this": rep, "other": old}, indent=2, default=str))
    (out / f"variety_{stamp}.md").write_text(markdown(rep, args.title, old))
    print(f"{rep['samples']} samples. Saved {out / f'variety_{stamp}.md'}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
