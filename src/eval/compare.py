"""Compare two eval runs question by question: what changed, and did it help?

Usage (from the repo root):
    python -m eval.compare results/<baseline>.json results/<experiment>.json

Prints:
  1. config differences (check that an experiment changed one thing only)
  2. each metric for B with its change from A, overall and per category
  3. questions whose outcome flipped (answer verdict, or abstention for unanswerable)
  4. questions whose retrieval recall changed
  5. full-price cost per run of each (as if nothing were cached)

With ~35 questions, one question is ~3 points. Trust the flip list more than
small metric deltas: a change that fixes its target questions and breaks none
is a win; +3 points made of one fix and two random flips is noise.
"""

import argparse
import json
import sys
from pathlib import Path

METRICS = [("hit_at_k", "hit@k"), ("recall_at_k", "recall"), ("mrr", "MRR"), ("correct", "correct"),
           ("false_abstention", "f.abst"), ("correct_abstention", "abstain"), ("citation_supports", "cite.ok")]
LOWER_IS_BETTER = {"false_abstention"}
OUTCOME_RANK = {"incorrect": 0, "partial": 1, "correct": 2, "answered": 0, "abstained": 2}


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, name + "."))
        else:
            out[name] = value
    return out


def config_diff(a: dict, b: dict) -> list[str]:
    fa, fb = flatten(a), flatten(b)
    return [f"{k}: {fa.get(k, '(unset)')} -> {fb.get(k, '(unset)')}"
            for k in sorted(fa.keys() | fb.keys()) if fa.get(k) != fb.get(k)]


def outcome(q: dict) -> str:
    m = q["metrics"]
    if q["category"] == "unanswerable":
        return "abstained" if m["abstained"] else "answered"
    return m["verdict"]


def cell(b, a, key) -> str:
    if b is None:
        return "-"
    if a is None:
        return f"{b:.2f}"
    d = b - a
    sign = "" if abs(d) < 0.005 else ("+" if d > 0 else "")
    good = d < 0 if key in LOWER_IS_BETTER else d > 0
    mark = "" if abs(d) < 0.005 else (" ▲" if good else " ▼")
    return f"{b:.2f} ({sign}{d:.2f}){mark}" if abs(d) >= 0.005 else f"{b:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", help="reference run (e.g. the baseline)")
    parser.add_argument("b", help="run to compare against it")
    args = parser.parse_args()
    A, B = (json.loads(Path(p).read_text()) for p in (args.a, args.b))

    if (A["run"]["corpus"], A["run"]["split"]) != (B["run"]["corpus"], B["run"]["split"]):
        print("warning: runs are on different corpora or splits", file=sys.stderr)
    qa = {q["id"]: q for q in A["questions"]}
    qb = {q["id"]: q for q in B["questions"]}
    if qa.keys() != qb.keys():
        print(f"warning: question sets differ ({len(qa)} vs {len(qb)}); comparing the overlap", file=sys.stderr)

    print("CONFIG CHANGES (A -> B)")
    for line in config_diff(A["config"], B["config"]) or ["(none)"]:
        print(f"  {line}")

    print(f"\nSCORES: B (change vs A)    ▲ better  ▼ worse")
    print(f"{'':14} " + "".join(f"{label:>16}" for _, label in METRICS))
    rows = [("OVERALL", A["summary"]["overall"], B["summary"]["overall"])]
    rows += [(c, A["summary"]["by_category"].get(c, {}), agg) for c, agg in B["summary"]["by_category"].items()]
    for name, sa, sb in rows:
        print(f"{name:14} " + "".join(f"{cell(sb.get(k), sa.get(k), k):>16}" for k, _ in METRICS))

    better, worse = [], []
    for qid in sorted(qa.keys() & qb.keys()):
        oa, ob = outcome(qa[qid]), outcome(qb[qid])
        if oa != ob:
            line = f"  {qid}  {qb[qid]['category']:12} {oa} -> {ob}"
            (better if OUTCOME_RANK[ob] > OUTCOME_RANK[oa] else worse).append(line)
    print(f"\nANSWER FLIPS: {len(better)} better, {len(worse)} worse")
    for line in better:
        print(line.replace("  dl", "+ dl", 1))
    for line in worse:
        print(line.replace("  dl", "- dl", 1))

    print("\nRETRIEVAL CHANGES (recall@k)")
    changed = [(qid, qa[qid]["metrics"]["recall"], qb[qid]["metrics"]["recall"]) for qid in sorted(qa.keys() & qb.keys())
               if "recall" in qa[qid]["metrics"] and qa[qid]["metrics"]["recall"] != qb[qid]["metrics"]["recall"]]
    for qid, ra, rb in changed:
        print(f"  {'+' if rb > ra else '-'} {qid}  {qb[qid]['category']:12} {ra:.2f} -> {rb:.2f}")
    if not changed:
        print("  (none)")

    ca, cb = A["cost"]["usd_if_uncached"], B["cost"]["usd_if_uncached"]
    print(f"\nFULL-PRICE COST PER RUN: A ${ca:.4f}  B ${cb:.4f}"
          + (f"  ({cb / ca:.1f}x)" if ca else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
