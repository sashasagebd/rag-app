"""Notebook tests. Pure checks run anywhere; the database tests skip if Postgres isn't up.
No network and no real models: embeddings, token counts and the LLM are faked."""

import tomllib
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from notebook import ask as ask_module
from notebook.fetch import FetchError, resolve_public
from notebook.ingest import add_bytes, check_document, name_from_url, safe_name, sniff_type
from notebook.store import NotebookError

LIMITS = {"max_file_mb": 1, "max_pdf_pages": 2}


def make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    return doc.tobytes()


# ---------- file checks ----------

def test_sniff_type_uses_content_not_extension():
    assert sniff_type(make_pdf(["hi"])) == "pdf"
    assert sniff_type(b"  <!DOCTYPE html><html><body>x</body></html>") == "html"
    assert sniff_type(b"<html><p>hi</p></html>") == "html"
    assert sniff_type(b"PK\x03\x04 a zip file") is None
    assert sniff_type(b"just some text") is None


def test_check_document_limits():
    with pytest.raises(NotebookError, match="limit is 1 MB"):
        check_document(b"%PDF-" + b"x" * 1_100_000, LIMITS)
    with pytest.raises(NotebookError, match="only PDF and HTML"):
        check_document(b"hello", LIMITS)
    with pytest.raises(NotebookError, match="3 pages; the limit is 2"):
        check_document(make_pdf(["a", "b", "c"]), LIMITS)
    with pytest.raises(NotebookError, match="damaged"):
        check_document(b"%PDF-1.7 not really a pdf", LIMITS)
    assert check_document(make_pdf(["a"]), LIMITS) == "pdf"


def test_safe_names():
    assert safe_name("../../etc/passwd") == "passwd"
    assert safe_name("C:\\Users\\me\\notes#1.pdf") == "notes1.pdf"
    assert safe_name("   ") == "document"
    assert name_from_url("https://arxiv.org/pdf/1706.03762", "pdf") == "1706.03762.pdf"
    assert name_from_url("https://example.org/", "html") == "example.org.html"
    assert name_from_url("https://example.org/a/page.html", "html") == "page.html"


# ---------- URL safety ----------

@pytest.mark.parametrize("url", [
    "http://127.0.0.1/", "http://localhost:8080/", "http://10.0.0.5/", "http://192.168.1.10:8080/admin",
    "http://169.254.169.254/latest/meta-data", "http://[::1]/", "http://[::ffff:127.0.0.1]/", "http://0.0.0.0/",
])
def test_private_addresses_are_refused(url):
    with pytest.raises(FetchError, match="private or local"):
        resolve_public(url)


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.org/x", "javascript:alert(1)"])
def test_only_http_is_allowed(url):
    with pytest.raises(FetchError, match="only http"):
        resolve_public(url)


def test_public_ip_literal_is_allowed():
    assert resolve_public("https://1.1.1.1/x") == ("https", 443, "1.1.1.1")


# ---------- database: add, dedupe, replace, remove, ask routing ----------

psycopg = pytest.importorskip("psycopg")
from ingest.embed_load import connect, database_url  # noqa: E402
from notebook import store  # noqa: E402

CONFIG = tomllib.loads((Path(__file__).parent.parent / "configs/default.toml").read_text())
CONFIG.update({
    "chunking": {"method": "pack_sections", "max_tokens": 50},
    "notebook": {**LIMITS, "full_context_max_tokens": 1000},
    "answer": {"model": "fake", "effort": "low", "max_tokens": 10, "prompt": "v2"},
})
NB = "pytest-nb"
FAKE = {"count_tokens": lambda t: len(t.split()), "embed_fn": lambda texts: np.ones((len(texts), 384), np.float32)}


@pytest.fixture
def conn(tmp_path):
    try:
        c = connect(database_url(CONFIG))
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable; run `docker compose up -d --wait`")
    CONFIG["notebook"]["storage_dir"] = str(tmp_path)
    c.execute("DELETE FROM collections WHERE name = %s", (NB,))
    store.create_notebook(c, NB)
    yield c
    c.execute("DELETE FROM collections WHERE name = %s", (NB,))
    c.close()


HTML = b"<html><body><h1>Intro</h1><p>" + b"The grading uses quizzes and homework. " * 5 + b"</p></body></html>"


def test_add_dedupe_replace_remove(conn, tmp_path):
    r = add_bytes(conn, NB, HTML, "page.html", source="page.html", config=CONFIG, **FAKE)
    assert r["status"] == "added" and r["n_chunks"] >= 1
    assert (tmp_path / NB / "page.html").read_bytes() == HTML

    again = add_bytes(conn, NB, HTML, "copy.html", source="copy.html", config=CONFIG, **FAKE)
    assert again["status"] == "skipped" and "page.html" in again["reason"]

    changed = HTML.replace(b"quizzes", b"exams")
    assert add_bytes(conn, NB, changed, "page.html", source="page.html", config=CONFIG, **FAKE)["status"] == "replaced"
    nid = store.get_notebook(conn, NB)
    assert [d[0] for d in store.list_documents(conn, nid)] == ["page.html"]

    store.remove_document(conn, nid, "page.html")
    assert conn.execute("SELECT count(*) FROM chunks WHERE collection_id = %s", (nid,)).fetchone()[0] == 0
    with pytest.raises(NotebookError, match="no document named"):
        store.remove_document(conn, nid, "page.html")


def test_scanned_pdf_is_rejected_with_a_clear_error(conn):
    with pytest.raises(NotebookError, match="OCR"):
        add_bytes(conn, NB, make_pdf(["", ""]), "scan.pdf", source="scan.pdf", config=CONFIG, **FAKE)


def test_corpus_collections_are_not_notebooks(conn):
    load = pytest.importorskip("ingest.embed_load").load_collection
    load(conn, "pytest-corpus", [], [])
    try:
        with pytest.raises(NotebookError, match="eval corpus"):
            store.get_notebook(conn, "pytest-corpus")
        with pytest.raises(NotebookError, match="already exists"):
            store.create_notebook(conn, "pytest-corpus")
    finally:
        conn.execute("DELETE FROM collections WHERE name = 'pytest-corpus'")


def test_small_notebook_sends_everything_large_uses_retrieval(conn, monkeypatch):
    add_bytes(conn, NB, HTML, "page.html", source="page.html", config=CONFIG, **FAKE)
    seen = {}

    def fake_answer(question, chunks, **kw):
        seen["chunks"] = chunks
        return {"data": {"found": True, "answer": "ok", "citations": [chunks[0]["chunk_key"]]},
                "usage": {"input_tokens": 1, "output_tokens": 1}, "cached": False}

    monkeypatch.setattr(ask_module, "answer_question", fake_answer)
    r = ask_module.ask(conn, NB, "How is it graded?", CONFIG)
    assert r["route"] == "full" and r["cited_chunks"][0]["doc"] == "page.html"
    n_all = len(seen["chunks"])

    monkeypatch.setattr(ask_module, "search_from_config", lambda *a, **k: seen["chunks"][:1])
    small_limit = {**CONFIG, "notebook": {**CONFIG["notebook"], "full_context_max_tokens": 1}}
    r = ask_module.ask(conn, NB, "How is it graded?", small_limit)
    assert r["route"] == "retrieval" and r["n_chunks_sent"] == 1 <= n_all
