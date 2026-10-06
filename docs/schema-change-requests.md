# Schema change requests — from the AI/RAG owner to the DB/Data owner

Consolidated list of everything the ingest pipeline needs from the schema.
The full corrected SQL is in `migrations/001_schema.sql`, where every change
is marked `[AI-RAG CHANGE]` with its reason inline.

---

## 🔴 Blocking — nothing can be ingested until this lands

### 1. `ai_content_vectors.vector` : `VECTOR(1536)` → `VECTOR(1024)`

pgvector's `vector(n)` declares an **exact** width, not a maximum. The chosen
embedding model (BGE-M3) emits 1024 dimensions, so every insert into a
`vector(1536)` column is rejected outright.

This is not a preference. No strong local multilingual embedding model emits
1536 dimensions — 1024 is the standard width for this model class (BGE-M3,
multilingual-e5-large, Qwen3-Embedding-0.6B are all 1024; EmbeddingGemma is
768). The original 1536 came from a cloud OpenAI/Gemini assumption that no
longer applies now that embedding runs locally.

Worth knowing for the record: pgvector's HNSW index supports at most **2000**
dimensions for the `vector` type. Anything wider needs `halfvec` (4000 max).
1024 is comfortably inside the limit.

---

## 🟡 Required for the agreed workflow

### 2. `posts` — director approval gate

```sql
approval_status VARCHAR(50) NOT NULL DEFAULT 'pending',
approved_at     TIMESTAMPTZ,
approved_by     BIGINT REFERENCES users(id) ON DELETE SET NULL,
CONSTRAINT chk_posts_approval_status
    CHECK (approval_status IN ('pending', 'approved', 'rejected'))
```

The agreed workflow requires super-admin/director approval **before**
verification and AI ingest. The original schema has no column recording that
this happened, so the gate cannot be enforced or audited at all. The pipeline
currently refuses to ingest anything that is not both `approved` and
`verified`.

### 3. `ai_content.chunk_index INTEGER NOT NULL`

Without it you cannot order chunks for the prompt, cite a position in an
answer, reassemble a document, or detect a botched chunking run. Paired with:

```sql
CONSTRAINT unique_post_chunk UNIQUE (post_id, chunk_index)
```

which makes duplicate-chunk corruption impossible to insert rather than merely
unlikely.

### 4. `ai_content_vectors` — one embedding per chunk per model

```sql
CONSTRAINT unique_content_model UNIQUE (ai_content_id, model_name)
```

Keeps a re-run from stacking duplicate vectors for the same chunk.

---

## 🟢 Recommended

| Change | Reason |
|---|---|
| `ai_content.token_count INTEGER NOT NULL` | The only way to verify chunk sizing behaves — especially for Bangla, which expands to far more subword tokens per character than English. |
| `posts.claimed_at TIMESTAMPTZ` | `'AI training in progress'` has no lease. A worker that dies mid-run leaves the row stuck in that state forever with nothing able to reclaim it. |
| `posts.content_hash VARCHAR(64)` | Lets the pipeline skip re-embedding when a re-publish did not actually change the body. Essential once edit-triggered re-ingest is built. |
| `fail_history.fail_chunk_index INTEGER` | "Embedding failed" with no chunk index is undebuggable once a post has 40 chunks. |
| All `TIMESTAMP` → `TIMESTAMPTZ` | Admission deadlines are time-sensitive and the team is at UTC+6. Naive timestamps shift silently when the server timezone differs. |
| Partial HNSW index per model | A single HNSW index over vectors from mixed models forces pgvector to post-filter on `model_name`, which can silently return fewer than K rows. See `migrations/001_schema.sql`. |

---

## Not requested, but flagged

- **`users.password`** is a bare `VARCHAR(255)` with no role or permission
  column, yet the workflow depends on a director-level approval role and the
  test plan includes authentication testing. Whoever owns auth needs to know
  there is currently no way to express "this user may approve".
- **No `conversations` table**, though the architecture document lists
  conversations as stored in PostgreSQL. Not needed by ingest; the chat slice
  will need it.
- **No structured admission tables** (programs, fees, deadlines). This is why
  the router's SQL branch was dropped — it would have nothing to query. If
  fee questions later prove to be answered badly from prose, that is the
  signal to add them.
