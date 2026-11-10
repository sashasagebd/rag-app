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
