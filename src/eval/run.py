"""Run the eval: retrieve, answer and judge every question in one split, then score.

Usage (from the repo root):
    python -m eval.run --corpus dl-slides --split dev --config configs/baseline.toml
    python -m eval.run --corpus dl-slides --split dev --judge-model claude-opus-5-5   # re-judge
    python -m eval.run --corpus dl-slides --split dev --limit 3                       # quick smoke test

The collection searched is the one named like the corpus. Writes
results/<timestamp>_<corpus>_<split>[_<tag>].json with the git commit, the full
merged config, cost, aggregate scores (overall and per category) and every
question's retrieved chunks, answer and verdict.

All embeddings and LLM calls go through the cache: re-running with nothing
changed costs $0 and gives identical results. Change the judge model and only
the judge calls are new.
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from answer.answer import answer_question
from common.cache import Cache
from common.config import load_config
from common.llm import cost_usd
from eval.judge import judge
from eval.metrics import score_question, summarize
from ingest.embed_load import connect, database_url
from retrieval.search import search


def git_info() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def load_questions(questions_dir: str, corpus: str, split: str) -> list[dict]:
    qs = []
    for path in sorted(Path(questions_dir).glob("*.jsonl")):
        qs += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [q for q in qs if q["corpus"] == corpus and q["split"] == split]


class CostTracker:
    """Sums tokens and dollars per role; 'spent' counts only calls that missed the cache."""

    def __init__(self, pricing: dict):
        self.pricing, self.roles = pricing, {}

    def add(self, role: str, model: str, result: dict) -> None:
        r = self.roles.setdefault(role, {"model": model, "calls": 0, "cached": 0, "input_tokens": 0,
                                         "output_tokens": 0, "usd_spent": 0.0, "usd_if_uncached": 0.0})
        usd = cost_usd(result["usage"], model, self.pricing)
        r["calls"] += 1
        r["cached"] += result["cached"]
        r["input_tokens"] += result["usage"]["input_tokens"]
        r["output_tokens"] += result["usage"]["output_tokens"]
        r["usd_if_uncached"] += usd
        if not result["cached"]:
            r["usd_spent"] += usd

    def total(self, key: str) -> float:
        return round(sum(r[key] for r in self.roles.values()), 4)


def fmt(value) -> str:
    return "  -  " if value is None else f"{value:5.2f}"


def print_summary(summary: dict, k: int) -> None:
    cols = [("hit_at_k", f"hit@{k}"), ("recall_at_k", f"rec@{k}"), ("mrr", "MRR"), ("correct", "correct"),
            ("partial", "partial"), ("false_abstention", "f.abst"), ("correct_abstention", "abstain"),
            ("citation_supports", "cite.ok")]
    print(f"\n{'':14} {'n':>3}  " + "  ".join(f"{label:>7}" for _, label in cols))
    rows = [("OVERALL", summary["overall"])] + list(summary["by_category"].items())
    for name, agg in rows:
        n = agg["n_answerable"] + agg["n_unanswerable"]
        print(f"{name:14} {n:3}  " + "  ".join(f"{fmt(agg[key]):>7}" for key, _ in cols))
    print("\nhit/rec/MRR: retrieval · correct/partial: judge · f.abst: said 'not found' but answer existed"
          "\nabstain: said 'not found' on unanswerable (higher is better) · cite.ok: a citation covers a source")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--split", required=True, choices=["dev", "test"])
    parser.add_argument("--config", default="configs/baseline.toml")
    parser.add_argument("--judge-model", help="override [judge].model")
    parser.add_argument("--limit", type=int, help="only the first N questions (smoke test)")
    parser.add_argument("--tag", help="suffix for the results file name")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.judge_model:
        config["judge"]["model"] = args.judge_model
    for role in ("answer", "judge"):
        if config[role]["model"] not in config["pricing"]:
            print(f"no [pricing] entry for {config[role]['model']}", file=sys.stderr)
            return 1

    questions = load_questions(config["paths"]["questions_dir"], args.corpus, args.split)[: args.limit]
    if not questions:
        print(f"no questions for corpus={args.corpus} split={args.split}", file=sys.stderr)
        return 1

    k, emb_model = config["retrieval"]["k"], config["embedding"]["model"]
    alternates = config.get("eval", {}).get("alternate_sources", {})
    cache = Cache(config["paths"]["cache_db"])
    costs = CostTracker(config["pricing"])
    records = []

    with connect(database_url(config)) as conn:
        for i, q in enumerate(questions, 1):
            chunks = search(conn, args.corpus, q["question"], k, emb_model, cache)
            if not chunks:
                print(f"collection '{args.corpus}' is empty: run ingest.run first", file=sys.stderr)
                return 1

            a_cfg = config["answer"]
            ans = answer_question(q["question"], chunks, model=a_cfg["model"], effort=a_cfg["effort"],
                                  max_tokens=a_cfg["max_tokens"], cache=cache)
            costs.add("answer", a_cfg["model"], ans)

            verdict = reason = None
            if q["category"] != "unanswerable" and ans["data"]["found"]:
                j_cfg = config["judge"]
                j = judge(q["question"], q["expected_answer"], ans["data"]["answer"], q["category"],
                          model=j_cfg["model"], effort=j_cfg["effort"], max_tokens=j_cfg["max_tokens"], cache=cache)
                costs.add("judge", j_cfg["model"], j)
                verdict, reason = j["data"]["verdict"], j["data"]["reason"]

            metrics = score_question(q, chunks, ans["data"], verdict, alternates)
            records.append({
                **q,
                "retrieved": [{"chunk_key": c["chunk_key"], "sections": c["sections"],
                               "distance": round(c["distance"], 4)} for c in chunks],
                "answer": ans["data"],
                "judge": {"model": config["judge"]["model"], "verdict": verdict, "reason": reason},
                "metrics": metrics,
            })
            shown = metrics.get("verdict") or ("abstained" if metrics["abstained"] else "answered")
            hit = f"hit={'y' if metrics['hit'] else 'n'}" if "hit" in metrics else "     "
            print(f"[{i:2}/{len(questions)}] {q['id']}  {q['category']:12} {hit}  {shown}")

    summary = summarize(records)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = Path(config["paths"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stamp}_{args.corpus}_{args.split}{'_' + args.tag if args.tag else ''}.json"
    result = {
        "run": {"timestamp": stamp, "corpus": args.corpus, "split": args.split, "config_file": args.config,
                "limit": args.limit, "git": git_info()},
        "config": config,
        "cost": {"usd_spent": costs.total("usd_spent"), "usd_if_uncached": costs.total("usd_if_uncached"),
                 "by_role": costs.roles},
        "summary": summary,
        "questions": records,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print_summary(summary, k)
    print(f"\ncost: ${result['cost']['usd_spent']:.4f} spent this run "
          f"(${result['cost']['usd_if_uncached']:.4f} without the cache)")
    print(f"results: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
