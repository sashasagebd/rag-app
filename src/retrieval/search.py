"""Vector search: the top-k chunks of one collection for a question.

Usage (from the repo root), to see what retrieval returns:
    python -m retrieval.search "how is the course graded?" --collection dl-slides

The question is embedded with the bge query prefix and compared with every
chunk by cosine distance (pgvector's <=> operator). Lower distance = closer.
No index yet: ~600 chunks is an exact scan in milliseconds.
"""

import argparse
import sys

import psycopg

from common.cache import Cache
from common.config import load_config
from common.embedding import QUERY_PREFIX, embed
from ingest.embed_load import connect, database_url


def search(conn: psycopg.Connection, collection: str, question: str, k: int,
           model_name: str, cache: Cache | None = None) -> list[dict]:
    """Return up to k chunks, closest first, each with its cosine distance."""
    [query_vec] = embed([QUERY_PREFIX + question], model_name, cache=cache)
    rows = conn.execute(
        """
        SELECT ch.chunk_key, ch.doc, ch.sections, ch.text, ch.embedding <=> %s AS distance
        FROM chunks ch JOIN collections co ON co.id = ch.collection_id
        WHERE co.name = %s
        ORDER BY distance
        LIMIT %s
        """,
        (query_vec, collection, k),
    ).fetchall()
    return [
        {"chunk_key": r[0], "doc": r[1], "sections": r[2], "text": r[3], "distance": float(r[4])}
        for r in rows
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--collection", required=True)
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--config", default="configs/default.toml")
    args = parser.parse_args()

    config = load_config(args.config)
    with connect(database_url(config)) as conn:
        hits = search(conn, args.collection, args.question, args.k, config["embedding"]["model"],
                      Cache(config["paths"]["cache_db"]))
    if not hits:
        print(f"no chunks in collection '{args.collection}'", file=sys.stderr)
        return 1
    for rank, h in enumerate(hits, 1):
        print(f"{rank}. {h['chunk_key']}  distance {h['distance']:.3f}")
        for s in h["sections"]:
            print(f"     {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
