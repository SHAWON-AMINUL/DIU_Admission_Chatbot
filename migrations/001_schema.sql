-- ============================================================
-- DIU ADMISSION DATABASE
-- PostgreSQL + pgvector + RAG
--
-- Baseline schema (DB/Data owner's slice) PLUS the changes the
-- AI/RAG ingest pipeline requires. Every deviation from the
-- original hand-off is marked  [AI-RAG CHANGE]  with a reason.
--
-- Hand this file to the DB/Data owner. See docs/schema-change-requests.md
-- ============================================================


-- ============================================================
-- 1. EXTENSIONS
-- ============================================================

CREATE EXTENSION IF NOT EXISTS vector;


-- ============================================================
-- 2. USERS
-- ============================================================

CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(150) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    post_count INTEGER NOT NULL DEFAULT 0,

    -- [AI-RAG CHANGE] TIMESTAMP -> TIMESTAMPTZ.
    -- Reason: admission deadlines are time-sensitive and the team is in
    -- UTC+6. Naive timestamps silently shift when the server TZ differs.
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);


-- ============================================================
-- 3. CATEGORIES
-- ============================================================

CREATE TABLE IF NOT EXISTS categories (
    id BIGSERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by BIGINT,

    CONSTRAINT fk_categories_created_by
        FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL
);


-- ============================================================
-- 4. POSTS
-- ============================================================

CREATE TABLE IF NOT EXISTS posts (
    id BIGSERIAL PRIMARY KEY,
    title VARCHAR(500) NOT NULL,
    body TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by BIGINT NOT NULL,
    published_at TIMESTAMPTZ,
    status VARCHAR(50) NOT NULL DEFAULT 'Draft',
    fail_count INTEGER NOT NULL DEFAULT 0,

    verify_status VARCHAR(50),
    verified_at TIMESTAMPTZ,
    verified_by BIGINT,

    -- [AI-RAG CHANGE] NEW: director-level approval gate.
    -- Reason: the agreed workflow requires super-admin/director approval
    -- BEFORE verification and AI ingest. The original schema had no way to
    -- record that this happened, so the gate could not be enforced at all.
    approval_status VARCHAR(50) NOT NULL DEFAULT 'pending',
    approved_at TIMESTAMPTZ,
    approved_by BIGINT,

    -- [AI-RAG CHANGE] NEW: worker lease.
    -- Reason: 'AI training in progress' with no lease means a worker that
    -- crashes mid-run leaves the row stuck in that state forever, with
    -- nothing able to reclaim it.
    claimed_at TIMESTAMPTZ,

    -- [AI-RAG CHANGE] NEW: content fingerprint.
    -- Reason: lets the pipeline skip re-embedding when a re-publish did not
    -- actually change the body. Cheap now, essential once edits are handled.
    content_hash VARCHAR(64),

    deleted_at TIMESTAMPTZ,
    deleted_by BIGINT,

    CONSTRAINT fk_posts_created_by
        FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE RESTRICT,
    CONSTRAINT fk_posts_verified_by
        FOREIGN KEY (verified_by) REFERENCES users(id) ON DELETE SET NULL,
    CONSTRAINT fk_posts_approved_by
        FOREIGN KEY (approved_by) REFERENCES users(id) ON DELETE SET NULL,
    CONSTRAINT fk_posts_deleted_by
        FOREIGN KEY (deleted_by) REFERENCES users(id) ON DELETE SET NULL,

    CONSTRAINT chk_posts_status
        CHECK (status IN (
            'Draft', 'published', 'verified',
            'AI training in progress', 'AI training complete',
            'AI training failed', 'deleted'
        )),

    -- [AI-RAG CHANGE] NEW: constrain the approval vocabulary.
    CONSTRAINT chk_posts_approval_status
        CHECK (approval_status IN ('pending', 'approved', 'rejected'))
);


-- ============================================================
-- 5. POST FILES
-- Attachment ingestion is OUT OF SCOPE for the MVP (agreed).
-- Table kept so the editorial side can still store attachments.
-- ============================================================

CREATE TABLE IF NOT EXISTS post_files (
    id BIGSERIAL PRIMARY KEY,
    post_id BIGINT NOT NULL,
    title VARCHAR(255),
    path TEXT NOT NULL,
    file_type VARCHAR(100),

    CONSTRAINT fk_post_files_post
        FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE
);


-- ============================================================
-- 6. FAIL HISTORY
-- ============================================================

CREATE TABLE IF NOT EXISTS fail_history (
    id BIGSERIAL PRIMARY KEY,
    post_id BIGINT NOT NULL,
    fail_reason TEXT,

    -- [AI-RAG CHANGE] NEW: which chunk failed.
    -- Reason: "embedding failed" with no chunk index is undebuggable once a
    -- post has 40 chunks. NULL means the failure was not chunk-specific.
    fail_chunk_index INTEGER,

    fail_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_fail_history_post
        FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE
);


-- ============================================================
-- 7. CATEGORY_POST  (many-to-many: posts <-> categories)
-- ============================================================

CREATE TABLE IF NOT EXISTS category_post (
    id BIGSERIAL PRIMARY KEY,
    category_id BIGINT NOT NULL,
    post_id BIGINT NOT NULL,
    created_by BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_category_post_category
        FOREIGN KEY (category_id) REFERENCES categories(id) ON DELETE CASCADE,
    CONSTRAINT fk_category_post_post
        FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE,
    CONSTRAINT fk_category_post_created_by
        FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL,
    CONSTRAINT unique_category_post UNIQUE (category_id, post_id)
);


-- ============================================================
-- 8. AI_CONTENT  (RAG chunks)
-- ============================================================

CREATE TABLE IF NOT EXISTS ai_content (
    id BIGSERIAL PRIMARY KEY,
    post_id BIGINT NOT NULL,

    -- Text actually sent to the embedding model, including the title
    -- prefix. Stored verbatim so retrieval results are reproducible.
    chunk_text TEXT NOT NULL,

    -- [AI-RAG CHANGE] NEW: ordinal position of this chunk within its post.
    -- Reason: without it you cannot cite position, reassemble a document,
    -- order context for the prompt, or detect a botched chunking run.
    chunk_index INTEGER NOT NULL,

    -- [AI-RAG CHANGE] NEW: BGE-M3 subword token count.
    -- Reason: the only way to verify chunk sizing is behaving, especially
    -- for Bangla, which expands to far more subword tokens per character.
    token_count INTEGER NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Lexical/full-text retrieval vector.
    -- KNOWN LIMITATION (accepted for MVP): the 'english' text search config
    -- does not stem or usefully tokenize Bangla, so this contributes
    -- nothing for Bangla queries. Bangla retrieval is vector-only until
    -- BGE-M3 sparse weights are added in a later phase.
    search_vector TSVECTOR
        GENERATED ALWAYS AS (
            to_tsvector('english', COALESCE(chunk_text, ''))
        ) STORED,

    CONSTRAINT fk_ai_content_post
        FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE,

    -- [AI-RAG CHANGE] NEW: one row per (post, chunk position).
    -- Reason: makes duplicate-chunk corruption impossible to insert rather
    -- than merely unlikely.
    CONSTRAINT unique_post_chunk UNIQUE (post_id, chunk_index)
);


-- ============================================================
-- 9. AI_CONTENT_VECTORS  (embeddings)
-- ============================================================

CREATE TABLE IF NOT EXISTS ai_content_vectors (
    id BIGSERIAL PRIMARY KEY,
    ai_content_id BIGINT NOT NULL,

    -- Model identity INCLUDING dimension, e.g. 'bge-m3@1024'.
    model_name VARCHAR(255) NOT NULL,

    -- [AI-RAG CHANGE] *** URGENT / BLOCKING ***  vector(1536) -> vector(1024)
    -- Reason: pgvector's vector(n) is an EXACT width, not a maximum. BGE-M3
    -- emits 1024 dimensions, so every INSERT into a vector(1536) column is
    -- rejected outright. Nothing can be ingested until this is changed.
    -- (No strong local multilingual model emits 1536; 1024 is the standard.)
    vector VECTOR(1024) NOT NULL,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_ai_vectors_ai_content
        FOREIGN KEY (ai_content_id) REFERENCES ai_content(id) ON DELETE CASCADE,

    -- [AI-RAG CHANGE] NEW: one embedding per chunk per model.
    -- Reason: makes re-embedding idempotent and keeps a retry from stacking
    -- duplicate vectors for the same chunk.
    CONSTRAINT unique_content_model UNIQUE (ai_content_id, model_name)
);


-- ============================================================
-- 10. INDEXES
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_users_email               ON users(email);
CREATE INDEX IF NOT EXISTS idx_posts_created_by          ON posts(created_by);
CREATE INDEX IF NOT EXISTS idx_posts_status              ON posts(status);
CREATE INDEX IF NOT EXISTS idx_posts_created_at          ON posts(created_at);
CREATE INDEX IF NOT EXISTS idx_post_files_post_id        ON post_files(post_id);
CREATE INDEX IF NOT EXISTS idx_fail_history_post_id      ON fail_history(post_id);
CREATE INDEX IF NOT EXISTS idx_category_post_category_id ON category_post(category_id);
CREATE INDEX IF NOT EXISTS idx_category_post_post_id     ON category_post(post_id);
CREATE INDEX IF NOT EXISTS idx_ai_content_post_id        ON ai_content(post_id);

-- Full text search
CREATE INDEX IF NOT EXISTS idx_ai_content_search_vector
    ON ai_content USING GIN(search_vector);

-- [AI-RAG CHANGE] Cosine HNSW index is now PARTIAL, per model.
-- Reason: a single index over vectors from mixed models forces pgvector to
-- post-filter on model_name, which can silently return fewer than K rows.
-- One partial index per model keeps every probe inside a single model's
-- vector space. Add another when a second model is introduced.
CREATE INDEX IF NOT EXISTS idx_ai_content_vectors_cosine_bge_m3
    ON ai_content_vectors USING HNSW (vector vector_cosine_ops)
    WHERE model_name = 'bge-m3@1024';

CREATE INDEX IF NOT EXISTS idx_ai_content_vectors_model   ON ai_content_vectors(model_name);
CREATE INDEX IF NOT EXISTS idx_ai_content_vectors_content ON ai_content_vectors(ai_content_id);


-- ============================================================
-- DONE
-- ============================================================
