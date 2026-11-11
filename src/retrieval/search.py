"""Retrieval: the top-k chunks of one collection for a question.

Usage (from the repo root), to see what retrieval returns:
    python -m retrieval.search "how is the course graded?" --collection dl-slides
    python -m retrieval.search "..." --collection dl-slides --config configs/exp-e4-hybrid.toml

Methods ([retrieval] method):
  vector  embed the question (bge query prefix) and rank chunks by cosine
          distance (pgvector's <=>). Lower distance = closer.
  full    every chunk of the collection, in document order (small notebooks:
          the whole thing fits in the prompt, so nothing is left to chance).
  hybrid  also run Postgres keyword search (full-text, any word may match,
          ranked by ts_rank_cd) and merge the two rankings with reciprocal
          rank fusion: score = sum over lists of 1 / (60 + rank). Exact terms
          like "LeNet" or "61,706" then count even when the embedding misses them.

Optional reranking ([retrieval] rerank_model): fetch rerank_candidates chunks
with the method above, score each (question, chunk) pair with a local
cross-encoder, and keep the k best. Slower (one model pass per pair, cached),
usually more precise.

No vector index yet: a few hundred chunks is an exact scan in milliseconds.
"""

import argparse
import sys

import psycopg

from common.cache import Cache
from common.config import load_config
from common.embedding import MODEL_LOCK, QUERY_PREFIX, embed
from ingest.embed_load import connect, database_url

RRF_K = 60  # standard reciprocal-rank-fusion constant

VECTOR_SQL = """
SELECT ch.chunk_key, ch.doc, ch.sections, ch.text, ch.pages, ch.embedding <=> %(qvec)s AS distance
FROM chunks ch JOIN collections co ON co.id = ch.collection_id
WHERE co.name = %(collection)s
ORDER BY distance
LIMIT %(limit)s
"""

HYBRID_SQL = """
WITH base AS (
    SELECT ch.* FROM chunks ch JOIN collections co ON co.id = ch.collection_id
    WHERE co.name = %(collection)s
),
query AS (  -- the question's words, OR-ed: any word may match
    SELECT replace(plainto_tsquery('english', %(question)s)::text, '&', '|')::tsquery AS t
),
vec AS (
    SELECT id, row_number() OVER (ORDER BY embedding <=> %(qvec)s) AS rank
    FROM base ORDER BY embedding <=> %(qvec)s LIMIT %(pool)s
),
kw AS (
    SELECT id, row_number() OVER (ORDER BY ts_rank_cd(tsv, query.t) DESC) AS rank
    FROM base, query WHERE tsv @@ query.t
    ORDER BY ts_rank_cd(tsv, query.t) DESC LIMIT %(pool)s
),
fused AS (
    SELECT id, sum(1.0 / (%(rrf_k)s + rank)) AS score
    FROM (SELECT * FROM vec UNION ALL SELECT * FROM kw) ranked
    GROUP BY id
)
SELECT b.chunk_key, b.doc, b.sections, b.text, b.pages, b.embedding <=> %(qvec)s AS distance
FROM fused f JOIN base b ON b.id = f.id
ORDER BY f.score DESC
LIMIT %(limit)s
"""


FULL_SQL = """
SELECT ch.chunk_key, ch.doc, ch.sections, ch.text, ch.pages, NULL::float AS distance
FROM chunks ch JOIN collections co ON co.id = ch.collection_id
WHERE co.name = %(collection)s
ORDER BY ch.doc, split_part(ch.chunk_key, '#', 2)::int
"""


def retrieve(conn: psycopg.Connection, collection: str, question: str, qvec, limit: int,
             method: str = "vector") -> list[dict]:
    """The database step: given the question and its embedding, the top `limit` chunks.
    method "full" ignores the question and returns every chunk, in document order."""
    sql = {"vector": VECTOR_SQL, "hybrid": HYBRID_SQL, "full": FULL_SQL}.get(method)
    if sql is None:
        raise ValueError(f"unknown retrieval method: {method}")
    params = {"qvec": qvec, "collection": collection, "question": question, "limit": limit,
              "pool": max(limit, 20), "rrf_k": RRF_K}
    return [{"chunk_key": r[0], "doc": r[1], "sections": r[2], "text": r[3], "pages": r[4],
             "distance": None if r[5] is None else float(r[5])}
            for r in conn.execute(sql, params).fetchall()]


_rerankers: dict = {}


def get_reranker(name: str):
    """Load a cross-encoder once per process (under the shared model lock)."""
    with MODEL_LOCK:
        if name not in _rerankers:
            from sentence_transformers import CrossEncoder  # slow import; only when needed

            _rerankers[name] = CrossEncoder(name)
        return _rerankers[name]


def rerank(question: str, chunks: list[dict], model_name: str, k: int, cache: Cache | None = None) -> list[dict]:
    """Re-order chunks by a cross-encoder's relevance score and keep the top k."""
    scores: list = [None] * len(chunks)
    missing = []
    for i, c in enumerate(chunks):
        hit = cache.get("rerank", [model_name, question, c["text"]]) if cache else None
        if hit is None:
            missing.append(i)
        else:
            scores[i] = hit
    if missing:
        new = get_reranker(model_name).predict([(question, chunks[i]["text"]) for i in missing])
        for i, s in zip(missing, new):
            scores[i] = float(s)
            if cache:
                cache.put("rerank", [model_name, question, chunks[i]["text"]], scores[i])
    order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:k]
    return [{**chunks[i], "rerank_score": round(scores[i], 4)} for i in order]


def search(conn: psycopg.Connection, collection: str, question: str, k: int, model_name: str,
           cache: Cache | None = None, method: str = "vector", rerank_model: str | None = None,
           rerank_candidates: int = 20) -> list[dict]:
    """Return up to k chunks, best first. Each has its cosine distance (and rerank_score if reranked).
    method "full" returns the whole collection instead: no embedding, no k, no reranking."""
    if method == "full":
        return retrieve(conn, collection, question, None, 0, "full")
    [qvec] = embed([QUERY_PREFIX + question], model_name, cache=cache)
    limit = max(k, rerank_candidates) if rerank_model else k
    chunks = retrieve(conn, collection, question, qvec, limit, method)
    if rerank_model:
        chunks = rerank(question, chunks, rerank_model, k, cache)
    return chunks


def search_from_config(conn, config: dict, collection: str, question: str, cache: Cache | None = None) -> list[dict]:
    r = config["retrieval"]
    return search(conn, collection, question, r["k"], config["embedding"]["model"], cache,
                  method=r.get("method", "vector"), rerank_model=r.get("rerank_model"),
                  rerank_candidates=r.get("rerank_candidates", 20))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--collection", help="default: [retrieval].collection, else required")
    parser.add_argument("-k", type=int, help="override [retrieval].k")
    parser.add_argument("--config", default="configs/baseline.toml")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.k:
        config["retrieval"]["k"] = args.k
    collection = args.collection or config["retrieval"].get("collection")
    if not collection:
        print("--collection is required", file=sys.stderr)
        return 1
    with connect(database_url(config)) as conn:
        hits = search_from_config(conn, config, collection, args.question, Cache(config["paths"]["cache_db"]))
    if not hits:
        print(f"no chunks in collection '{collection}'", file=sys.stderr)
        return 1
    for rank, h in enumerate(hits, 1):
        extra = f"  rerank {h['rerank_score']:.3f}" if "rerank_score" in h else ""
        print(f"{rank}. {h['chunk_key']}  distance {h['distance']:.3f}{extra}")
        for s in h["sections"]:
            print(f"     {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
