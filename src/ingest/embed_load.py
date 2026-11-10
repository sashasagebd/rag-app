"""Embed chunks and load them into Postgres + pgvector as a named collection.

Usage (from the repo root):
    python -m ingest.embed_load --collection dl-slides    # reads data/chunks/dl-slides.jsonl

Loading a collection REPLACES its chunks (delete + insert in one transaction),
so re-running after a re-chunk never leaves old and new chunks mixed.

Embeddings come from common/embedding.py (normalized, cached per text), so
re-loading unchanged chunks skips the model entirely.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from common.cache import Cache
from common.config import load_config
from common.embedding import embed

SCHEMA = Path(__file__).resolve().parents[2] / "db" / "schema.sql"


def database_url(config: dict) -> str:
    return os.environ.get("DATABASE_URL", config["database"]["url"])


def connect(url: str) -> psycopg.Connection:
    """Open a connection, make sure the tables exist, and teach psycopg the vector type."""
    conn = psycopg.connect(url, autocommit=True)
    conn.execute(SCHEMA.read_text())
    register_vector(conn)
    return conn


def load_collection(conn: psycopg.Connection, name: str, chunks: list[dict], embeddings) -> int:
    """Replace the collection's chunks with these. Returns the collection id."""
    assert len(chunks) == len(embeddings), "one embedding per chunk"
    with conn.transaction():
        collection_id = conn.execute(
            "INSERT INTO collections (name) VALUES (%s) "
            "ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id",
            (name,),
        ).fetchone()[0]
        conn.execute("DELETE FROM chunks WHERE collection_id = %s", (collection_id,))
        insert_chunks(conn, collection_id, chunks, embeddings)
    return collection_id


def insert_chunks(conn: psycopg.Connection, collection_id: int, chunks: list[dict], embeddings) -> None:
    """Insert chunk rows (call inside a transaction)."""
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (collection_id, chunk_key, doc, sections, pages, text, n_tokens, chunker, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
            [
                (collection_id, c["chunk_key"], c["doc"], Jsonb(c["sections"]), Jsonb(c.get("pages", [])), c["text"],
                 c["n_tokens"], c["chunker"], emb)
                for c, emb in zip(chunks, embeddings)
            ],
        )


def read_chunks(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def embed_and_load(chunks: list[dict], collection: str, config: dict) -> None:
    emb_cfg = config["embedding"]
    cache = Cache(config["paths"]["cache_db"])
    embeddings = embed([c["text"] for c in chunks], emb_cfg["model"], emb_cfg["batch_size"],
                       cache=cache, progress=True)
    if embeddings.shape[1] != emb_cfg["dim"]:
        raise ValueError(f"model gives {embeddings.shape[1]}-dim vectors, config/schema expect {emb_cfg['dim']}")
    with connect(database_url(config)) as conn:
        collection_id = load_collection(conn, collection, chunks, embeddings)
    print(f"loaded {len(chunks)} chunks into collection '{collection}' (id {collection_id})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--chunks", help="chunks JSONL (default: <chunks_dir>/<collection>.jsonl)")
    parser.add_argument("--config", default="configs/default.toml")
    args = parser.parse_args()

    config = load_config(args.config)
    path = Path(args.chunks or Path(config["paths"]["chunks_dir"]) / f"{args.collection}.jsonl")
    chunks = read_chunks(path)
    if not chunks:
        print(f"no chunks in {path}", file=sys.stderr)
        return 1
    embed_and_load(chunks, args.collection, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
