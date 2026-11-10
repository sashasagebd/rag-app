"""Loads fake embeddings into the docker Postgres. Skipped if the DB isn't running.

Needs `docker compose up -d --wait` first. Uses a throwaway collection and deletes it.
"""

import tomllib
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
psycopg = pytest.importorskip("psycopg")
pytest.importorskip("pgvector")

from ingest.embed_load import connect, database_url, load_collection  # noqa: E402

CONFIG = tomllib.loads((Path(__file__).parent.parent / "configs/default.toml").read_text())
DIM = CONFIG["embedding"]["dim"]
NAME = "pytest-tmp"


@pytest.fixture
def conn():
    try:
        c = connect(database_url(CONFIG))
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable; run `docker compose up -d --wait`")
    yield c
    c.execute("DELETE FROM collections WHERE name = %s", (NAME,))  # cascades to chunks
    c.close()


def chunk(key):
    return {"chunk_key": key, "doc": "d.html", "sections": ["A > B"], "text": key, "n_tokens": 1, "chunker": "test"}


def unit(i):
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def test_load_then_nearest_neighbour(conn):
    cid = load_collection(conn, NAME, [chunk("c0"), chunk("c1")], [unit(0), unit(1)])
    rows = conn.execute(
        "SELECT chunk_key, sections FROM chunks WHERE collection_id = %s "
        "ORDER BY embedding <=> %s LIMIT 1",  # <=> is cosine distance
        (cid, unit(1)),
    ).fetchall()
    assert rows == [("c1", ["A > B"])]


def test_reload_replaces_the_collection(conn):
    cid = load_collection(conn, NAME, [chunk("c0"), chunk("c1")], [unit(0), unit(1)])
    assert load_collection(conn, NAME, [chunk("new")], [unit(2)]) == cid
    keys = [r[0] for r in conn.execute("SELECT chunk_key FROM chunks WHERE collection_id = %s", (cid,))]
    assert keys == ["new"]


def test_hybrid_finds_a_keyword_match_the_vector_ranks_last(conn):
    from retrieval.search import retrieve

    chunks = [dict(chunk(f"c{i}"), text=t) for i, t in enumerate(
        ["attention scores and softmax", "residual blocks add the input", "LeNet-5 has 61,706 parameters"])]
    # The query vector is closest to c0, then c1; c2 (the keyword match) is farthest.
    q = unit(0) + 0.5 * unit(1)
    load_collection(conn, NAME, chunks, [unit(0), unit(1), unit(2)])
    vector = [c["chunk_key"] for c in retrieve(conn, NAME, "How many parameters does LeNet have?", q, 2, "vector")]
    hybrid = [c["chunk_key"] for c in retrieve(conn, NAME, "How many parameters does LeNet have?", q, 2, "hybrid")]
    assert vector == ["c0", "c1"]
    assert "c2" in hybrid
