-- Applied by src/ingest/embed_load.py before every load. Safe to rerun.

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
    embedding     vector(384) NOT NULL, -- dim of BAAI/bge-small-en-v1.5; change with the model
    UNIQUE (collection_id, chunk_key)
);
