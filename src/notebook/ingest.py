"""Add one document (an upload's bytes, or a URL) to a notebook.

    add_bytes(conn, "ml-papers", data, "attention.pdf", source="attention.pdf", config=config)
    add_url(conn, "ml-papers", "https://arxiv.org/pdf/1706.03762", config=config)

Steps, each with a clear error (NotebookError / FetchError) when it fails:
  1. size limit                     [notebook] max_file_mb
  2. file type from the CONTENT     PDF must start with %PDF; else it must look like HTML
  3. PDF checks                     opens, not encrypted, at most max_pdf_pages
  4. duplicate check                same bytes already in the notebook -> skipped
  5. parse -> must contain text     (scanned PDFs have none: OCR isn't supported)
  6. chunk + embed                  with the pipeline's [chunking] / [embedding] settings
  7. store                          replaces an older document of the same name
  8. keep the original file         [notebook] storage_dir/<notebook>/<name>
"""

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import psycopg
import pymupdf

from common.embedding import embed
from ingest.chunk import chunk_docs, make_token_counter
from ingest.parse import parse_bytes
from notebook import store
from notebook.fetch import fetch
from notebook.store import NotebookError

MIN_TEXT_CHARS = 50


def sniff_type(data: bytes) -> str | None:
    """'pdf', 'html' or None, from the first bytes (extensions can lie)."""
    head = data[:1024]
    if head.lstrip().startswith(b"%PDF") or b"%PDF-" in head:
        return "pdf"
    lowered = head.lstrip().lower()
    if any(tag in lowered for tag in (b"<!doctype html", b"<html", b"<head", b"<body")):
        return "html"
    return None


def safe_name(name: str) -> str:
    """A document name safe for display, file storage and chunk keys ('#' separates the chunk number)."""
    name = Path(name.replace("\\", "/")).name
    name = re.sub(r"[\x00-\x1f#]", "", name).strip() or "document"
    return name[:120]


def name_from_url(url: str, doc_type: str) -> str:
    parts = urlsplit(url)
    last = parts.path.rstrip("/").rsplit("/", 1)[-1] or parts.hostname or "page"
    name = safe_name(last)
    ext = ".pdf" if doc_type == "pdf" else ".html"
    return name if name.lower().endswith((ext, ".htm")) else name + ext


@lru_cache(maxsize=4)
def _token_counter(model_name: str):
    return make_token_counter(model_name)


def check_document(data: bytes, limits: dict) -> str:
    """Steps 1-3. Returns the doc type or raises NotebookError."""
    max_bytes = limits["max_file_mb"] * 1_000_000
    if len(data) > max_bytes:
        raise NotebookError(f"the file is {len(data) / 1e6:.1f} MB; the limit is {limits['max_file_mb']} MB")
    if not data:
        raise NotebookError("the file is empty")
    doc_type = sniff_type(data)
    if doc_type is None:
        raise NotebookError("only PDF and HTML documents are supported (the file's content is neither)")
    if doc_type == "pdf":
        try:
            with pymupdf.open(stream=data, filetype="pdf") as pdf:
                if pdf.needs_pass:
                    raise NotebookError("the PDF is password-protected")
                if pdf.page_count > limits["max_pdf_pages"]:
                    raise NotebookError(f"the PDF has {pdf.page_count} pages; the limit is {limits['max_pdf_pages']}")
        except NotebookError:
            raise
        except Exception:
            raise NotebookError("could not read this PDF (it may be damaged)")
    return doc_type


def add_bytes(conn: psycopg.Connection, notebook: str, data: bytes, name: str, *, source: str, config: dict,
              cache=None, count_tokens=None, embed_fn=None) -> dict:
    """Add one document. Returns {"status": "added"|"replaced"|"skipped", "name", ...}.
    count_tokens / embed_fn replace the real tokenizer / embedding model (tests)."""
    notebook_id = store.get_notebook(conn, notebook)
    limits = config["notebook"]
    name = safe_name(name)
    doc_type = check_document(data, limits)

    sha256 = hashlib.sha256(data).hexdigest()
    same = store.document_with_hash(conn, notebook_id, sha256)
    if same:
        return {"status": "skipped", "name": same, "reason": f"the same file is already in the notebook as '{same}'"}

    try:
        parsed = parse_bytes(data, name, doc_type)
    except Exception:
        raise NotebookError(f"could not extract text from this {doc_type.upper()}")
    if sum(len(s["text"]) for s in parsed["sections"]) < MIN_TEXT_CHARS:
        raise NotebookError(
            "no extractable text: scanned PDFs need OCR, which isn't supported" if doc_type == "pdf"
            else "no readable text (the page may build its content with JavaScript)")

    ch, emb = config["chunking"], config["embedding"]
    count_tokens = count_tokens or _token_counter(emb["model"])
    chunks = chunk_docs([parsed], ch["method"], ch["max_tokens"], count_tokens, ch.get("clean_latex", False))
    texts = [c["text"] for c in chunks]
    embeddings = embed_fn(texts) if embed_fn else embed(texts, emb["model"], emb["batch_size"], cache=cache)

    replaced = store.document_exists(conn, notebook_id, name)
    store.add_document(conn, notebook_id, {"name": name, "source": source, "doc_type": doc_type,
                                           "sha256": sha256, "n_bytes": len(data)}, chunks, embeddings)
    folder = Path(limits["storage_dir"]) / notebook
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_bytes(data)

    return {"status": "replaced" if replaced else "added", "name": name, "doc_type": doc_type,
            "n_chunks": len(chunks), "n_tokens": sum(c["n_tokens"] for c in chunks),
            "n_pages": max((p for c in chunks for p in c["pages"]), default=None)}


def add_url(conn: psycopg.Connection, notebook: str, url: str, *, config: dict, cache=None, **kwargs) -> dict:
    """Download (see fetch.py for the safety rules), then add_bytes."""
    store.get_notebook(conn, notebook)  # fail before downloading if the notebook doesn't exist
    limits = config["notebook"]
    data, final_url, _ = fetch(url, max_bytes=limits["max_file_mb"] * 1_000_000,
                               timeout=limits["url_timeout_s"], max_redirects=limits["max_redirects"])
    doc_type = sniff_type(data) or "html"
    return add_bytes(conn, notebook, data, name_from_url(final_url, doc_type), source=url, config=config,
                     cache=cache, **kwargs)
