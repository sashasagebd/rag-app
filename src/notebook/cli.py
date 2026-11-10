"""Notebooks from the command line: your own documents, cited answers.

Usage (from the repo root):
    python -m notebook.cli create ml-papers
    python -m notebook.cli add ml-papers paper.pdf https://example.org/page.html
    python -m notebook.cli list
    python -m notebook.cli docs ml-papers
    python -m notebook.cli ask ml-papers "What does the paper propose?" [--show-sources]
    python -m notebook.cli remove ml-papers paper.pdf
    python -m notebook.cli delete ml-papers          # asks you to type the name to confirm

Settings (limits, models, retrieval) come from configs/notebook.toml.
"""

import argparse
import shutil
import sys
import textwrap
from pathlib import Path

from common.cache import Cache
from common.config import load_config
from common.llm import cost_usd
from ingest.embed_load import connect, database_url
from notebook import store
from notebook.ask import ask
from notebook.fetch import FetchError
from notebook.ingest import add_bytes, add_url
from notebook.store import NotebookError


def where(chunk: dict) -> str:
    """'paper.pdf, p. 3-4' or 'slides.html, Intro > Grading' for a citation line."""
    pages = chunk.get("pages") or []
    if pages:
        span = str(pages[0]) if len(pages) == 1 else f"{pages[0]}-{pages[-1]}"
        return f"{chunk['doc']}, p. {span}"
    named = [s for s in chunk["sections"] if s]
    return f"{chunk['doc']}, {named[0]}" if named else chunk["doc"]


def cmd_add(conn, args, config, cache) -> int:
    failures = 0
    for item in args.items:
        try:
            if item.startswith(("http://", "https://")):
                print(f"downloading {item} ...", flush=True)
                r = add_url(conn, args.notebook, item, config=config, cache=cache)
            else:
                path = Path(item)
                if not path.is_file():
                    raise NotebookError(f"no such file: {item}")
                r = add_bytes(conn, args.notebook, path.read_bytes(), path.name, source=path.name,
                              config=config, cache=cache)
        except (NotebookError, FetchError) as e:
            print(f"  ✗ {item}: {e}", file=sys.stderr)
            failures += 1
            continue
        if r["status"] == "skipped":
            print(f"  - {item}: skipped, {r['reason']}")
        else:
            pages = f", {r['n_pages']} pages" if r["n_pages"] else ""
            note = " (replaced the older version)" if r["status"] == "replaced" else ""
            print(f"  ✓ {r['name']}: {r['n_chunks']} chunks, {r['n_tokens']:,} tokens{pages}{note}")
    return 1 if failures else 0


def cmd_ask(conn, args, config, cache) -> int:
    r = ask(conn, args.notebook, args.question, config, cache)
    route = ("whole notebook sent" if r["route"] == "full"
             else f"{r['n_chunks_sent']} best chunks of {r['notebook_tokens']:,} tokens")
    print()
    print(textwrap.fill(r["answer"]["answer"], width=88))
    if r["cited_chunks"]:
        print("\nSources:")
        for i, c in enumerate(r["cited_chunks"], 1):
            print(f"  [{i}] {where(c)}")
            if args.show_sources:
                print(textwrap.indent(textwrap.shorten(c["text"].replace("\n", " "), 600), "      "))
    cost = 0.0 if r["cached"] else cost_usd(r["usage"], config["answer"]["model"], config["pricing"])
    print(f"\n({route}; {config['answer']['model']}; ${cost:.4f})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/notebook.toml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("create").add_argument("notebook")
    sub.add_parser("list")
    sub.add_parser("docs").add_argument("notebook")
    p = sub.add_parser("add")
    p.add_argument("notebook")
    p.add_argument("items", nargs="+", help="PDF/HTML files or http(s) links")
    p = sub.add_parser("ask")
    p.add_argument("notebook")
    p.add_argument("question")
    p.add_argument("--show-sources", action="store_true", help="print each cited chunk's text")
    p = sub.add_parser("remove")
    p.add_argument("notebook")
    p.add_argument("document")
    p = sub.add_parser("delete")
    p.add_argument("notebook")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = parser.parse_args()

    config = load_config(args.config)
    cache = Cache(config["paths"]["cache_db"])
    try:
        with connect(database_url(config)) as conn:
            if args.command == "create":
                store.create_notebook(conn, args.notebook)
                print(f"created notebook '{args.notebook}'")
            elif args.command == "list":
                rows = store.list_notebooks(conn)
                for name, n_docs, tokens, created in rows:
                    print(f"{name:30} {n_docs:3} docs  {tokens:>9,} tokens  created {created:%Y-%m-%d}")
                if not rows:
                    print("no notebooks yet: python -m notebook.cli create <name>")
            elif args.command == "docs":
                rows = store.list_documents(conn, store.get_notebook(conn, args.notebook))
                for name, source, doc_type, n_bytes, n_chunks, tokens, added in rows:
                    src = f"  from {source}" if source != name else ""
                    print(f"{name:40} {doc_type:4} {n_bytes / 1e6:6.2f} MB {n_chunks:4} chunks "
                          f"{tokens:>8,} tokens{src}")
                if not rows:
                    print("no documents yet")
            elif args.command == "add":
                return cmd_add(conn, args, config, cache)
            elif args.command == "ask":
                return cmd_ask(conn, args, config, cache)
            elif args.command == "remove":
                store.remove_document(conn, store.get_notebook(conn, args.notebook), args.document)
                (Path(config["notebook"]["storage_dir"]) / args.notebook / args.document).unlink(missing_ok=True)
                print(f"removed '{args.document}'")
            elif args.command == "delete":
                notebook_id = store.get_notebook(conn, args.notebook)
                if not args.yes and input(f"Delete '{args.notebook}' and all its documents? "
                                          f"Type the name to confirm: ").strip() != args.notebook:
                    print("not deleted")
                    return 1
                store.delete_notebook(conn, notebook_id)
                shutil.rmtree(Path(config["notebook"]["storage_dir"]) / args.notebook, ignore_errors=True)
                print(f"deleted notebook '{args.notebook}' and all its documents")
    except NotebookError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
