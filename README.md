# rag-app

A document Q&A app that answers **only** from the documents you provide, with citations, plus an evaluation harness that scores retrieval and answers against known-answer questions.

**Status:** foundation only. Parsing and the question format are built. Chunking, embedding, retrieval and answering are not built yet.

## Setup

Requires Python 3.11+ and Docker.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"        # sentence-transformers pulls in PyTorch (~2 GB)

cp .env.example .env
docker compose up -d --wait    # Postgres 17 + pgvector on localhost:5433
```

Check the database:

```bash
docker compose exec db psql -U rag -d rag -c "select extversion from pg_extension where extname='vector';"
```

The host port is **5433** so it doesn't clash with a local Postgres on 5432. `docker compose down` stops the database. `docker compose down -v` also deletes its data.

## Usage

Run every command from the repo root. Paths come from `configs/default.toml`.

**Parse documents.** Put PDF/HTML files in `data/raw/`, then run:

```bash
python -m ingest.parse                     # everything in data/raw/
python -m ingest.parse data/raw/foo.pdf    # one file
```

Each input becomes `data/parsed/<filename>.json`, a list of sections with `page`, `heading_path` and `text`. See the docstring in `src/ingest/parse.py` for the exact rules.

**Validate eval questions:**

```bash
python -m eval.validate_questions          # every *.jsonl in data/questions/
```

**Run tests:**

```bash
pytest
```

## Eval question format

Each line of a `.jsonl` file in `data/questions/` holds one question:

| field | type | notes |
|---|---|---|
| `id` | string | unique across all files |
| `corpus` | string | which document set the question is about |
| `split` | `"dev"` \| `"test"` | tune on dev; report on test |
| `question` | string | |
| `expected_answer` | string \| null | `null` only for `unanswerable` |
| `source_refs` | list | `[{"doc": "handbook.pdf", "page": 12, "section": "Benefits > Leave"}]`. Each ref needs `doc` plus `page` and/or `section`. Empty only for `unanswerable` |
| `category` | enum | `single_fact`, `exact`, `multi_hop`, `unanswerable`, `adversarial` |

See `data/questions/examples.jsonl` for examples.

## Layout

```
configs/default.toml       paths (later: chunking/embedding/retrieval settings)
db/init.sql                enables pgvector on first DB start
data/raw/                  source documents (git-ignored)
data/parsed/               parse output, one JSON per document (git-ignored)
data/questions/            eval questions (JSONL)
src/ingest/parse.py        PDF/HTML -> sections JSON
src/retrieval/             (empty, not built yet)
src/eval/validate_questions.py
tests/
```

## Licence note

PDF parsing uses PyMuPDF, which is AGPL-licensed. That's fine for personal use. If the app is ever distributed commercially, switch to `pypdf`.
