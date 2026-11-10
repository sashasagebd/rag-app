"""Scoring: per-question metrics and their aggregates. Pure functions, no I/O.

Retrieval (answerable questions only):
  hit       any source_ref is covered by some retrieved chunk
  recall    fraction of source_refs covered (matters for multi_hop)
  rr        1 / rank of the first chunk covering any source_ref (0 if none) -> MRR

Answers:
  verdict            judge's verdict; an answerable question answered "not found" is "incorrect"
  abstained          the model said "not found"
  citations_valid    every cited chunk_key was among the retrieved chunks
  citation_supports  at least one cited chunk covers a source_ref

A chunk covers a source_ref when its doc matches and the ref's section is one
of the chunk's sections, or when the chunk's doc is an alternate source for
the ref's doc (config [eval.alternate_sources], e.g. the review deck).

Each source_refs entry is one needed source: a ref, or a list of alternative
refs (any one suffices). recall = covered entries / entries.
"""

from statistics import mean


def ref_groups(refs: list) -> list[list[dict]]:
    """Normalize source_refs to a list of alternative-groups: [ref, [r1, r2]] -> [[ref], [r1, r2]]."""
    return [entry if isinstance(entry, list) else [entry] for entry in refs]


def all_refs(refs: list) -> list[dict]:
    return [r for group in ref_groups(refs) for r in group]


def chunk_covers_ref(chunk: dict, ref: dict, alternates: dict) -> bool:
    if chunk["doc"] == ref["doc"]:
        if "section" in ref:
            return ref["section"] in chunk["sections"]
        return True  # page-only ref: chunks don't record pages yet (fine for this HTML corpus)
    return ref["doc"] in alternates.get(chunk["doc"], [])


def retrieval_scores(refs: list, chunks: list[dict], alternates: dict) -> dict:
    groups = ref_groups(refs)
    covered = [any(chunk_covers_ref(c, r, alternates) for c in chunks for r in g) for g in groups]
    flat = all_refs(refs)
    first = next(
        (rank for rank, c in enumerate(chunks, 1) if any(chunk_covers_ref(c, r, alternates) for r in flat)),
        None,
    )
    return {"hit": any(covered), "recall": sum(covered) / len(groups), "rr": 1 / first if first else 0.0}


def citation_scores(citations: list[str], chunks: list[dict], refs: list, alternates: dict) -> dict:
    retrieved = {c["chunk_key"]: c for c in chunks}
    flat = all_refs(refs)
    return {
        "citations_valid": all(k in retrieved for k in citations),
        "citation_supports": any(
            chunk_covers_ref(retrieved[k], r, alternates) for k in citations if k in retrieved for r in flat
        ),
    }


def score_question(question: dict, chunks: list[dict], answer: dict, verdict: str | None, alternates: dict) -> dict:
    """Metrics for one question. `answer` is {"found", "answer", "citations"}."""
    abstained = not answer["found"]
    m: dict = {"abstained": abstained}
    if question["category"] == "unanswerable":
        if not abstained:
            m["citations_valid"] = citation_scores(answer["citations"], chunks, [], alternates)["citations_valid"]
        return m
    m.update(retrieval_scores(question["source_refs"], chunks, alternates))
    m["verdict"] = "incorrect" if abstained else verdict
    if not abstained:
        m.update(citation_scores(answer["citations"], chunks, question["source_refs"], alternates))
    return m


def _avg(values: list) -> float | None:
    return round(mean(values), 3) if values else None


def aggregate(records: list[dict]) -> dict:
    """Average metrics over records [{"category": ..., "metrics": {...}}, ...]."""
    ans = [r["metrics"] for r in records if r["category"] != "unanswerable"]
    una = [r["metrics"] for r in records if r["category"] == "unanswerable"]
    answered = [m for m in ans if not m["abstained"]]
    with_cites = [m for m in ans + una if "citations_valid" in m]
    return {
        "n_answerable": len(ans),
        "n_unanswerable": len(una),
        "hit_at_k": _avg([m["hit"] for m in ans]),
        "recall_at_k": _avg([m["recall"] for m in ans]),
        "mrr": _avg([m["rr"] for m in ans]),
        "correct": _avg([m["verdict"] == "correct" for m in ans]),
        "partial": _avg([m["verdict"] == "partial" for m in ans]),
        "false_abstention": _avg([m["abstained"] for m in ans]),
        "correct_abstention": _avg([m["abstained"] for m in una]),
        "citations_valid": _avg([m["citations_valid"] for m in with_cites]),
        "citation_supports": _avg([m["citation_supports"] for m in answered]),
    }


def summarize(records: list[dict]) -> dict:
    categories = sorted({r["category"] for r in records})
    return {
        "overall": aggregate(records),
        "by_category": {c: aggregate([r for r in records if r["category"] == c]) for c in categories},
    }
