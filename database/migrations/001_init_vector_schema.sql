-- Enables pgvector's vector type and operators (cosine distance, etc.)
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    document_name TEXT NOT NULL,
    source TEXT NOT NULL,
    document_type TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'completed',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    page INTEGER,
    section TEXT,
    token_count INTEGER NOT NULL,
    -- Dimension must match the embedding model's output (BAAI/bge-small-en-v1.5 = 384).
    -- If EMBEDDING_MODEL ever changes to a model with a different output size,
    -- this column (and every stored embedding) must be migrated, not just the config.
    embedding vector(384) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);

-- Deliberately no ANN index (IVFFlat/HNSW) yet - see Phase 0 architecture doc,
-- risk #1. Brute-force search is fine at this corpus size; add an index once
-- real document volume justifies the added complexity and tuning burden.