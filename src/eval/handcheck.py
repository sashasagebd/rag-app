"""Hand-check judge verdicts: you read the answer and say whether the judge was right.

Usage (from the repo root):
    python -m eval.handcheck results/<run>.json
    python -m eval.handcheck results/<haiku-judged>.json --compare results/<opus-judged>.json

Shows up to N (default 10) judged answers one at a time. With --compare, cases
where the two judges disagree come first, since that's where judge errors hide.
Your answers are saved next to the run as <run>.handcheck.json.
"""

import argparse
import json
import random
import sys
from pathlib import Path

CHOICES = {"c": "correct", "p": "partial", "i": "incorrect"}


def pick(questions: list[dict], other: dict, n: int, seed: int) -> list[dict]:
    judged = [q for q in questions if q["judge"]["verdict"]]
    disagree = [q for q in judged if q["id"] in other and other[q["id"]] != q["judge"]["verdict"]]
    rest = [q for q in judged if q not in disagree]
    random.Random(seed).shuffle(rest)
    return (disagree + rest)[:n]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results")
    parser.add_argument("--compare", help="a second run of the same questions, judged by another model")
    parser.add_argument("-n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run = json.loads(Path(args.results).read_text())
    other = {}
    if args.compare:
        other_run = json.loads(Path(args.compare).read_text())
        other = {q["id"]: q["judge"]["verdict"] for q in other_run["questions"] if q["judge"]["verdict"]}
        n_both = sum(1 for q in run["questions"] if q["judge"]["verdict"] and q["id"] in other)
        n_agree = sum(1 for q in run["questions"] if q["judge"]["verdict"] and other.get(q["id"]) == q["judge"]["verdict"])
        print(f"judges agree on {n_agree}/{n_both} questions")

    checks = []
    for i, q in enumerate(pick(run["questions"], other, args.n, args.seed), 1):
        print("\n" + "=" * 72)
        print(f"[{i}] {q['id']}  ({q['category']})")
        print(f"\nQUESTION:  {q['question']}")
        print(f"EXPECTED:  {q['expected_answer']}")
        print(f"ANSWER:    {q['answer']['answer']}")
        print(f"\nJUDGE ({q['judge']['model']}): {q['judge']['verdict']} - {q['judge']['reason']}")
        if q["id"] in other:
            print(f"OTHER JUDGE: {other[q['id']]}")
        reply = input("\nIs the judge right? [y]es / [n]o / [s]kip / [q]uit: ").strip().lower()
        if reply.startswith("q"):
            break
        if reply.startswith("s") or not reply:
            continue
        mine = q["judge"]["verdict"]
        if reply.startswith("n"):
            while (c := input("What should it be? [c]orrect / [p]artial / [i]ncorrect: ").strip().lower()[:1]) not in CHOICES:
                pass
            mine = CHOICES[c]
        checks.append({"id": q["id"], "judge": q["judge"]["verdict"], "mine": mine,
                       "agree": mine == q["judge"]["verdict"]})

    if not checks:
        print("nothing recorded")
        return 0
    out = Path(args.results).with_suffix(".handcheck.json")
    out.write_text(json.dumps(checks, indent=2))
    agree = sum(c["agree"] for c in checks)
    print(f"\nyou agreed with the judge on {agree}/{len(checks)} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
