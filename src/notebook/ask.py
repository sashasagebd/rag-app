"""Ask a notebook a question: get a cited answer from its documents only.

    result = ask(conn, "ml-papers", "How many heads does the base model use?", config)
    result["answer"]        {"found", "answer", "citations"}
    result["route"]         "full" (small notebook: every chunk sent) or "retrieval"
    result["cited_chunks"]  the chunks the answer cites (doc, sections, pages, text)

Route: if the notebook's total size is at most [notebook] full_context_max_tokens,
the whole notebook goes to the model and nothing can be missed by search.
Otherwise the Phase 3 retrieval pipeline picks the chunks ([retrieval] settings).
Every query names one notebook, so it only ever sees that notebook's chunks.
"""

import psycopg

from answer.answer import answer_question
from notebook import store
from notebook.store import NotebookError
from retrieval.search import search, search_from_config


def ask(conn: psycopg.Connection, notebook: str, question: str, config: dict, cache=None) -> dict:
    notebook_id = store.get_notebook(conn, notebook)
    if not question.strip():
        raise NotebookError("the question is empty")
    tokens = store.notebook_tokens(conn, notebook_id)
    if tokens == 0:
        raise NotebookError(f"'{notebook}' has no documents yet: add one first")

    if tokens <= config["notebook"]["full_context_max_tokens"]:
        route = "full"
        chunks = search(conn, notebook, question, 0, config["embedding"]["model"], method="full")
    else:
        route = "retrieval"
        chunks = search_from_config(conn, config, notebook, question, cache)

    a = config["answer"]
    result = answer_question(question, chunks, model=a["model"], effort=a["effort"], max_tokens=a["max_tokens"],
                             cache=cache, prompt=a.get("prompt", "v1"))
    by_key = {c["chunk_key"]: c for c in chunks}
    cited = [by_key[k] for k in result["data"]["citations"] if k in by_key]
    return {"route": route, "notebook_tokens": tokens, "answer": result["data"], "cited_chunks": cited,
            "n_chunks_sent": len(chunks), "usage": result["usage"], "cached": result["cached"]}
