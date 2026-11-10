"""Ingest a folder into a named collection: parse -> chunk -> embed and load.

Usage (from the repo root):
    python -m ingest.run data/raw/notes --collection dl-slides

Each stage still writes its files (data/parsed/, data/chunks/), exactly as if
you had run ingest.parse, ingest.chunk and ingest.embed_load one by one.
If any file fails to parse, nothing is loaded.
"""

import argparse
import sys
from pathlib import Path

from common.config import load_config
from ingest.chunk import chunk_docs, make_token_counter, write_chunks
from ingest.embed_load import embed_and_load
from ingest.parse import collect_inputs, parse_file, write_parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="raw PDF/HTML files or directories")
    parser.add_argument("--collection", required=True, help="collection name, e.g. dl-slides")
    parser.add_argument("--config", default="configs/default.toml")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = config["paths"]

    # 1. parse
    inputs = collect_inputs(args.inputs)
    if not inputs:
        print("no PDF/HTML files found", file=sys.stderr)
        return 1
    parsed_dir = Path(paths["parsed_dir"])
    parsed_dir.mkdir(parents=True, exist_ok=True)
    docs, failures = [], []
    for path in inputs:
        try:
            doc = parse_file(path)
        except Exception as e:
            failures.append(f"{path}: {e}")
            continue
        write_parsed(doc, parsed_dir)
        docs.append(doc)
    if failures:
        print("parse failed, nothing loaded:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    print(f"1/3 parsed  {len(docs)} files -> {parsed_dir}/")

    # 2. chunk
    chunking = config["chunking"]
    count_tokens = make_token_counter(config["embedding"]["model"])
    chunks = chunk_docs(docs, chunking["method"], chunking["max_tokens"], count_tokens)
    chunks_path = Path(paths["chunks_dir"]) / f"{args.collection}.jsonl"
    write_chunks(chunks, chunks_path)
    print(f"2/3 chunked {len(chunks)} chunks ({chunking['method']}, max {chunking['max_tokens']} tokens) -> {chunks_path}")

    # 3. embed + load
    print("3/3 embedding and loading ...")
    embed_and_load(chunks, args.collection, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
