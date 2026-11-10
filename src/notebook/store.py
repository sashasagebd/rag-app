"""Notebook storage: every SQL statement that creates, lists or changes notebooks.

A notebook is a row in `collections` with kind = 'notebook'. Its documents are
rows in `documents`; their chunks are rows in `chunks` (doc = document name).
Eval corpora (kind = 'corpus') can't be changed from here.

user_id: notebooks made in the web app belong to a user (collections.user_id).
Functions that take user_id only see that user's notebooks; None (the CLI) sees all.
Documents have a status: 'processing' (web upload being embedded), 'ready', 'failed'.
"""

import re

import psycopg

from ingest.embed_load import insert_chunks

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")


class NotebookError(Exception):
    """A problem the user can fix; the message is safe to show."""


class NotFoundError(NotebookError):
    """The notebook or document doesn't exist (or isn't this user's)."""


def create_notebook(conn: psycopg.Connection, name: str, user_id: int | None = None) -> int:
    if not NAME_RE.match(name):
        raise NotebookError("notebook names use lowercase letters, digits and dashes (max 63), e.g. ml-papers")
    row = conn.execute("SELECT kind FROM collections WHERE name = %s", (name,)).fetchone()
    if row:
        raise NotebookError(f"'{name}' already exists" + (" (as an eval corpus)" if row[0] != "notebook" else ""))
    return conn.execute(
        "INSERT INTO collections (name, kind, user_id) VALUES (%s, 'notebook', %s) RETURNING id", (name, user_id)
    ).fetchone()[0]


def get_notebook(conn: psycopg.Connection, name: str, user_id: int | None = None) -> int:
    row = conn.execute("SELECT id, kind, user_id FROM collections WHERE name = %s", (name,)).fetchone()
    if not row or (user_id is not None and row[2] != user_id):
        raise NotFoundError(f"no notebook named '{name}'")  # same message: don't reveal others' notebooks
    if row[1] != "notebook":
        raise NotebookError(f"'{name}' is an eval corpus, not a notebook")
    return row[0]


def list_notebooks(conn: psycopg.Connection, user_id: int | None = None) -> list[tuple]:
    """(name, n_documents, n_tokens, created_at) for every notebook (of one user, if given)."""
    return conn.execute(
        "SELECT c.name, count(d.id), coalesce(sum(d.n_tokens), 0), c.created_at "
        "FROM collections c LEFT JOIN documents d ON d.collection_id = c.id "
        "WHERE c.kind = 'notebook' AND (%(uid)s::int IS NULL OR c.user_id = %(uid)s) "
        "GROUP BY c.id ORDER BY c.name", {"uid": user_id}
    ).fetchall()


def count_notebooks(conn: psycopg.Connection, user_id: int) -> int:
    return conn.execute("SELECT count(*) FROM collections WHERE kind = 'notebook' AND user_id = %s",
                        (user_id,)).fetchone()[0]


def user_upload_bytes(conn: psycopg.Connection, user_id: int) -> int:
    return conn.execute(
        "SELECT coalesce(sum(d.n_bytes), 0) FROM documents d JOIN collections c ON c.id = d.collection_id "
        "WHERE c.user_id = %s", (user_id,)).fetchone()[0]


def list_documents(conn: psycopg.Connection, notebook_id: int) -> list[tuple]:
    """(name, source, doc_type, n_bytes, n_chunks, n_tokens, added_at, status, error) per document."""
    return conn.execute(
        "SELECT name, source, doc_type, n_bytes, n_chunks, n_tokens, added_at, status, error FROM documents "
        "WHERE collection_id = %s ORDER BY added_at", (notebook_id,)
    ).fetchall()


def mark_processing(conn: psycopg.Connection, notebook_id: int, doc: dict) -> None:
    """Record a web upload before its background processing starts. A re-upload of an
    existing name keeps the old chunks searchable until the new ones replace them."""
    conn.execute(
        "INSERT INTO documents (collection_id, name, source, doc_type, sha256, n_bytes, n_chunks, n_tokens, status) "
        "VALUES (%s, %s, %s, %s, %s, %s, 0, 0, 'processing') "
        "ON CONFLICT (collection_id, name) DO UPDATE SET status = 'processing', error = NULL",
        (notebook_id, doc["name"], doc["source"], doc["doc_type"], doc["sha256"], doc["n_bytes"]),
    )


def mark_failed(conn: psycopg.Connection, notebook_id: int, name: str, error: str) -> None:
    conn.execute("UPDATE documents SET status = 'failed', error = %s WHERE collection_id = %s AND name = %s",
                 (error, notebook_id, name))


def fail_interrupted(conn: psycopg.Connection) -> int:
    """At startup: uploads that were processing when the server stopped will never finish."""
    return conn.execute("UPDATE documents SET status = 'failed', error = 'interrupted by a server restart; "
                        "please upload it again' WHERE status = 'processing'").rowcount


def document_with_hash(conn: psycopg.Connection, notebook_id: int, sha256: str) -> str | None:
    row = conn.execute("SELECT name FROM documents WHERE collection_id = %s AND sha256 = %s "
                       "AND status <> 'failed'", (notebook_id, sha256)).fetchone()  # a failed upload may be retried
    return row[0] if row else None


def document_exists(conn: psycopg.Connection, notebook_id: int, name: str) -> bool:
    return conn.execute("SELECT 1 FROM documents WHERE collection_id = %s AND name = %s",
                        (notebook_id, name)).fetchone() is not None


def add_document(conn: psycopg.Connection, notebook_id: int, doc: dict, chunks: list[dict], embeddings) -> None:
    """Add (or replace) one document and its chunks, all in one transaction.
    doc: {name, source, doc_type, sha256, n_bytes}."""
    with conn.transaction():
        conn.execute("DELETE FROM chunks WHERE collection_id = %s AND doc = %s", (notebook_id, doc["name"]))
        conn.execute(
            "INSERT INTO documents (collection_id, name, source, doc_type, sha256, n_bytes, n_chunks, n_tokens) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (collection_id, name) DO UPDATE SET source = EXCLUDED.source, "
            "doc_type = EXCLUDED.doc_type, sha256 = EXCLUDED.sha256, n_bytes = EXCLUDED.n_bytes, "
            "n_chunks = EXCLUDED.n_chunks, n_tokens = EXCLUDED.n_tokens, added_at = now(), "
            "status = 'ready', error = NULL",
            (notebook_id, doc["name"], doc["source"], doc["doc_type"], doc["sha256"], doc["n_bytes"],
             len(chunks), sum(c["n_tokens"] for c in chunks)),
        )
        insert_chunks(conn, notebook_id, chunks, embeddings)


def remove_document(conn: psycopg.Connection, notebook_id: int, name: str) -> None:
    with conn.transaction():
        conn.execute("DELETE FROM chunks WHERE collection_id = %s AND doc = %s", (notebook_id, name))
        if conn.execute("DELETE FROM documents WHERE collection_id = %s AND name = %s",
                        (notebook_id, name)).rowcount == 0:
            raise NotFoundError(f"no document named '{name}' in this notebook")


def delete_notebook(conn: psycopg.Connection, notebook_id: int) -> None:
    conn.execute("DELETE FROM collections WHERE id = %s", (notebook_id,))  # cascades to documents + chunks


def notebook_tokens(conn: psycopg.Connection, notebook_id: int) -> int:
    return conn.execute("SELECT coalesce(sum(n_tokens), 0) FROM documents WHERE collection_id = %s "
                        "AND status = 'ready'", (notebook_id,)).fetchone()[0]
