-- Applied on every connection (src/ingest/embed_load.py: connect). Safe to rerun.

CREATE EXTENSION IF NOT EXISTS vector;

-- A named set of documents, e.g. "dl-slides". Later: one per user notebook.
CREATE TABLE IF NOT EXISTS collections (
    id         serial PRIMARY KEY,
    name       text NOT NULL UNIQUE,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id            bigserial PRIMARY KEY,
    collection_id integer NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    chunk_key     text NOT NULL,        -- stable id, e.g. "dl-s01.html#3"; answers cite these
    doc           text NOT NULL,        -- source filename
    sections      jsonb NOT NULL,       -- heading paths covered, e.g. ["Course Introduction > Grading"]
    text          text NOT NULL,        -- exactly what was embedded
    n_tokens      integer NOT NULL,
    chunker       text NOT NULL,        -- e.g. "pack_sections:500", to tell experiments apart
    embedding     vector NOT NULL,      -- any size; each collection uses one model ([embedding].dim)
    UNIQUE (collection_id, chunk_key)
);

-- Migration for tables created with the old fixed vector(384) column. No-op once done.
DO $$
BEGIN
    IF (SELECT atttypmod FROM pg_attribute
        WHERE attrelid = 'chunks'::regclass AND attname = 'embedding') <> -1 THEN
        ALTER TABLE chunks ALTER COLUMN embedding TYPE vector;
    END IF;
END $$;

-- Keyword search (hybrid retrieval): Postgres full-text vector of the chunk text,
-- kept up to date by Postgres itself, with an index for fast matching.
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);

-- Notebooks (Phase 4). A notebook is a collection with kind = 'notebook'; corpora
-- built by ingest.run keep kind = 'corpus'. owner is filled in by Phase 5's login.
ALTER TABLE collections ADD COLUMN IF NOT EXISTS kind text NOT NULL DEFAULT 'corpus';
ALTER TABLE collections ADD COLUMN IF NOT EXISTS owner text;

-- One row per document added to a notebook. Its chunks are the rows of `chunks`
-- with the same collection_id and doc = documents.name.
CREATE TABLE IF NOT EXISTS documents (
    id            serial PRIMARY KEY,
    collection_id integer NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    name          text NOT NULL,          -- e.g. "attention.pdf"; chunks.doc
    source        text NOT NULL,          -- original filename or URL
    doc_type      text NOT NULL,          -- "pdf" or "html"
    sha256        text NOT NULL,          -- of the original bytes: detects duplicate uploads
    n_bytes       integer NOT NULL,
    n_chunks      integer NOT NULL,
    n_tokens      integer NOT NULL,       -- sum of chunk tokens: decides full-context vs retrieval
    added_at      timestamptz NOT NULL DEFAULT now(),
    UNIQUE (collection_id, name)
);

-- PDF pages each chunk covers (empty for HTML). Lets citations say "p. 4".
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS pages jsonb NOT NULL DEFAULT '[]';
