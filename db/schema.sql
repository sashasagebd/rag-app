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

-- ---------- Phase 5: web app accounts ----------

CREATE TABLE IF NOT EXISTS users (
    id                   serial PRIMARY KEY,
    username             text NOT NULL UNIQUE,
    password_hash        text NOT NULL,         -- scrypt, see src/api/security.py
    api_key_enc          text,                  -- the user's Anthropic key, encrypted with APP_SECRET_KEY
    api_key_hint         text,                  -- e.g. "sk-ant-…a1b2", safe to show
    use_server_key       boolean NOT NULL DEFAULT false,  -- admin-granted: may use the server's key
    daily_question_limit integer,               -- NULL = the [app] default
    created_at           timestamptz NOT NULL DEFAULT now()
);

-- One-time invite codes. Only a hash is stored: the code itself is shown once, to the admin.
CREATE TABLE IF NOT EXISTS invites (
    code_hash  text PRIMARY KEY,
    note       text,
    created_at timestamptz NOT NULL DEFAULT now(),
    used_by    integer REFERENCES users(id) ON DELETE SET NULL,
    used_at    timestamptz
);

-- Login sessions. The browser holds a random token in an HttpOnly cookie; only its hash is stored.
CREATE TABLE IF NOT EXISTS sessions (
    token_hash text PRIMARY KEY,
    user_id    integer NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL
);

-- Every question asked through the app: chat history, daily limits and spend.
CREATE TABLE IF NOT EXISTS questions (
    id            bigserial PRIMARY KEY,
    user_id       integer NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    collection_id integer NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    question      text NOT NULL,
    response      jsonb NOT NULL,       -- answer, citations (with chunk text), route
    model         text NOT NULL,
    key_source    text NOT NULL,        -- 'user', 'server' or 'cache'
    usd           numeric(10, 6) NOT NULL DEFAULT 0,
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS questions_user_day_idx ON questions (user_id, created_at);

-- Notebooks belong to a user (NULL = created from the CLI). Replaces the unused text `owner`.
ALTER TABLE collections DROP COLUMN IF EXISTS owner;
ALTER TABLE collections ADD COLUMN IF NOT EXISTS user_id integer REFERENCES users(id) ON DELETE CASCADE;

-- Web uploads are processed in the background: 'processing' -> 'ready' or 'failed' (with error).
ALTER TABLE documents ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'ready';
ALTER TABLE documents ADD COLUMN IF NOT EXISTS error text;
