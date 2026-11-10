"""Ask a notebook a question: get a cited answer from its documents only.

    result = ask(conn, notebook, "How many heads does the base model use?", config)
    result["answer"]        {"found", "answer", "citations"}
    result["route"]         "full" (small notebook: every chunk sent) or "retrieval"
    result["cited_chunks"]  the chunks the answer cites (doc, sections, pages, text)

Route: if the notebook's total size is at most [notebook] full_context_max_tokens,
the whole notebook goes to the model and nothing can be missed by search.
Otherwise the Phase 3 retrieval pipeline picks the chunks ([retrieval] settings).
Every query names one notebook, so it only ever sees that notebook's chunks.

`notebook` is the collection name; `notebook_id` (optional) skips the lookup when
the caller has already checked ownership (the web API does).
"""

import psycopg

from answer.answer import answer_question
from notebook import store
from notebook.store import NotebookError
from retrieval.search import search, search_from_config


def cite_label(chunk: dict) -> str:
    """'paper.pdf, p. 3-4' or 'slides.html, Intro > Grading': where a citation points."""
    pages = chunk.get("pages") or []
    if pages:
        span = str(pages[0]) if len(pages) == 1 else f"{pages[0]}-{pages[-1]}"
        return f"{chunk['doc']}, p. {span}"
    named = [s for s in chunk["sections"] if s]
    return f"{chunk['doc']}, {named[0]}" if named else chunk["doc"]


def ask(conn: psycopg.Connection, notebook: str, question: str, config: dict, cache=None, *,
        notebook_id: int | None = None, api_key: str | None = None) -> dict:
    if notebook_id is None:
        notebook_id = store.get_notebook(conn, notebook)
    if not question.strip():
        raise NotebookError("the question is empty")
    tokens = store.notebook_tokens(conn, notebook_id)
    if tokens == 0:
        raise NotebookError("this notebook has no ready documents yet: add one first")

    if tokens <= config["notebook"]["full_context_max_tokens"]:
        route = "full"
        chunks = search(conn, notebook, question, 0, config["embedding"]["model"], method="full")
    else:
        route = "retrieval"
        chunks = search_from_config(conn, config, notebook, question, cache)

    a = config["answer"]
    result = answer_question(question, chunks, model=a["model"], effort=a["effort"], max_tokens=a["max_tokens"],
                             cache=cache, prompt=a.get("prompt", "v1"), api_key=api_key)
    by_key = {c["chunk_key"]: c for c in chunks}
    cited = [by_key[k] for k in result["data"]["citations"] if k in by_key]
    return {"route": route, "notebook_tokens": tokens, "answer": result["data"], "cited_chunks": cited,
            "n_chunks_sent": len(chunks), "usage": result["usage"], "cached": result["cached"]}
