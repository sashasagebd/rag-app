# rag-app

A document Q&A app that answers **only** from the documents you provide, with citations, plus an evaluation harness that scores retrieval and answers against known-answer questions.

**Status:** ingestion, retrieval, cited answering (Claude API) and the eval harness are built. Next: experiments.

## Setup

Requires Python 3.11+ and Docker.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only PyTorch, skips ~2 GB of CUDA
pip install -e ".[dev]"

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

**Ingest a folder into a collection** (parse, chunk, embed, load into Postgres):

```bash
python -m ingest.run data/raw/notes --collection dl-slides
```

The first run downloads the embedding model (`BAAI/bge-small-en-v1.5`, ~130 MB). Re-running replaces the collection's chunks. Intermediate files land in `data/parsed/` and `data/chunks/<collection>.jsonl` so you can inspect each stage. Chunking and embedding settings are in `configs/default.toml`.

**Or run the stages one at a time.** Parse first: put PDF/HTML files in `data/raw/`, then run:

```bash
python -m ingest.parse                     # everything in data/raw/
python -m ingest.parse data/raw/foo.pdf    # one file
```

Each input becomes `data/parsed/<filename>.json`, a list of sections with `page`, `heading_path` and `text`. See the docstring in `src/ingest/parse.py` for the exact rules.

```bash
python -m ingest.chunk data/parsed --collection dl-slides   # -> data/chunks/dl-slides.jsonl
python -m ingest.embed_load --collection dl-slides          # -> Postgres
```

**Set up the API key** (answers and judging use the Claude API). Put `ANTHROPIC_API_KEY=...` in `.env`, then check it:

```bash
python -m common.llm
```

**Search, or run the eval:**

```bash
python -m retrieval.search "how is the course graded?" --collection dl-slides
python -m eval.run --corpus dl-slides --split dev --config configs/baseline.toml
python -m eval.handcheck results/<run>.json        # check the judge by hand
```

Each run writes `results/<timestamp>_<corpus>_<split>.json` and prints scores per question category. Embeddings and LLM responses are cached in `data/cache/`, so re-running unchanged settings is free.

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
| `source_refs` | list | `[{"doc": "handbook.pdf", "page": 12, "section": "Benefits > Leave"}]`. Each ref needs `doc` plus `page` and/or `section`. An entry may be a list of 2+ refs meaning "any one of these". Empty only for `unanswerable` |
| `category` | enum | `single_fact`, `exact`, `multi_hop`, `unanswerable`, `adversarial` |

See `data/questions/examples.jsonl` for examples.

## Layout

```
configs/default.toml       paths (later: chunking/embedding/retrieval settings)
db/init.sql                enables pgvector on first DB start
data/raw/                  source documents (git-ignored)
data/parsed/               parse output, one JSON per document (git-ignored)
data/chunks/               chunk output, one JSONL per collection (git-ignored)
data/questions/            eval questions (JSONL)
src/ingest/parse.py        PDF/HTML -> sections JSON
src/ingest/chunk.py        sections -> chunks (pluggable chunkers)
src/ingest/embed_load.py   chunks -> embeddings -> Postgres
src/ingest/run.py          all three in one command
src/common/                config, cache, embeddings, the one LLM call
src/retrieval/search.py    top-k vector search
src/answer/answer.py       cited answers from retrieved chunks
src/eval/run.py            the eval harness (judge.py, metrics.py, handcheck.py beside it)
configs/baseline.toml      the Phase 2 baseline settings
results/                   one JSON per eval run
db/schema.sql              collections + chunks tables
src/retrieval/             (empty, not built yet)
src/eval/validate_questions.py
tests/
```

## Licence note

PDF parsing uses PyMuPDF, which is AGPL-licensed. That's fine for personal use. If the app is ever distributed commercially, switch to `pypdf`.
