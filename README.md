<<<<<<< HEAD
# DIU Admission RAG

The **AI/RAG slice**: ingest verified admission posts into chunks and vectors,
retrieve them with two-way hybrid search, and answer applicant questions from
those retrieved chunks — as a grounded chat response or a plain search hit
list. Runs entirely locally — no cloud embedding API, no cloud generation API,
no API keys, no rate limits.

## Stack

| Piece | Choice | Why |
|---|---|---|
| Embeddings | **BGE-M3**, 1024-dim, local | Best multilingual/Bangla retrieval quality of the local options; symmetric (no query/passage prefixes to get wrong) |
| Vector store | PostgreSQL 18 + pgvector 0.8.6 | HNSW cosine index, partial per model |
| Lexical | Postgres full-text (`tsvector` + GIN) | Already in the schema |
| Fusion | Reciprocal Rank Fusion, k=60 | Rank-based, so cosine distance and `ts_rank` never need normalizing against each other |
| DB access | raw `psycopg3` | The schema is owned by a teammate; an ORM copy of it would drift |
| Python | 3.13 via `uv` | `sentence-transformers`/PyTorch do not support 3.14 |
| Generation | **gemma3:4b** via Ollama, local | Keeps the no-API-keys property. Chosen by probe: the alternative tested could not produce Bangla at all |
| Orchestration | `langchain-core` 1.x | Prompt/LLM/retriever composition and streaming only — retrieval stays hand-rolled |

## Setup

```bash
# 1. Database — Homebrew PostgreSQL 18 + pgvector, on port 5433.
#    Port 5433 is deliberate: it leaves an existing PostgreSQL on 5432
#    (an EnterpriseDB install, say) completely untouched.
brew install postgresql@18 pgvector
bash scripts/setup_db.sh

# 2. Dependencies
uv sync

# 3. Embedding service — first run downloads BGE-M3 (~2.3 GB), then works offline
uv run uvicorn app.embeddings.service:app --port 8001

# 4. Dashboard (separate terminal)
uv run uvicorn app.main:app --reload --port 8000

# 5. Generation — Ollama, running locally, then the model (~3.3 GB, offline after)
brew install ollama
ollama serve &   # or the Ollama menu-bar app
ollama pull gemma3:4b
```

Open <http://localhost:8000/chat> for the assistant, or
<http://localhost:8000> for the ingestion dashboard.

| Endpoint | Purpose |
|---|---|
| `POST /api/search` | retrieval only — the debugging surface |
| `POST /api/chat` | grounded answer plus sources |
| `POST /api/chat/stream` | the same, streamed as SSE |
| `GET /chat` | the chatbot UI |

## Run the pipeline

```bash
uv run python scripts/seed.py                      # 15 sample posts
uv run python -m app.ingestion.run --all-eligible  # chunk + embed + store
uv run pytest -v                                   # prove retrieval works
```

Single post: `uv run python -m app.ingestion.run --post-id 3`

Every run first reclaims posts stuck in `'AI training in progress'` past their
lease (`--reclaim-after MINUTES`, default 30). A worker killed mid-run — crash,
Ctrl-C, a machine restart — otherwise leaves its post claimed forever, neither
complete nor eligible, invisible to every later run.

### Verified on this machine

```
15 posts → 15 chunks → 15 vectors, all 'AI training complete'
pytest: 15 passed (10/10 question→post pairs in top-3, plus the Bangla query)
ingest of 15 posts: ~0.6s once the model is warm (M1 Pro, MPS)
```

## Workflow

```
Draft ──publish──> published ──approve──> approved ──verify──> verified
                                       (director)                 │
                                                                  ▼
                                              status = 'AI training in progress'
                                                                  │
                        chunk (BGE-M3 tokenizer, ~300 tok, 60 overlap, title-prefixed)
                                                                  │
                                                          ai_content rows
                                                                  │
                                            embed (local BGE-M3, L2-normalized)
                                                                  │
                                                    ai_content_vectors rows
                                                                  │
                          success ──> 'AI training complete'   failure ──> fail_history,
                                                                            fail_count += 1,
                                                                            'AI training failed'
```

Verification is the trigger. It runs as a background task so the verify
request never blocks on embedding — this is the seam the Backend teammate's
verify handler plugs into (it calls the same `ingest_post()`).

## Deliberate scope limits

These are decided, not overlooked:

- **Attachments are not ingested.** Body text only. `post_files` still stores
  them for the editorial side. This is the biggest content-coverage risk —
  if most real admission content arrives as PDF fee tables, the bot will say
  "I don't have that information" for the most-asked questions. Worth checking
  with admission staff before committing to the current plan.
- **No retry / resume / re-ingest-on-edit.** A failed run is parked for a
  human. The one exception is `DELETE FROM ai_content WHERE post_id = ?`
  before insert, so re-running never duplicates chunks.
- **No SQL/structured branch in the router.** There are no structured
  admission tables to query, so a third branch would query nothing.
- **Full-text search is English-only.** `to_tsvector('english', ...)` does not
  stem or usefully tokenize Bangla, so for a Bangla query the lexical branch
  contributes nothing and retrieval is vector-only. BGE-M3's cross-lingual
  embeddings make that workable — but **do not present this as working hybrid
  search in Bangla.** The fix is BGE-M3's sparse weights, in a later phase.

  This is measurable, not theoretical. Same question, two languages:

  | Query | Top hit | vector rank | text rank |
  |---|---|---|---|
  | "What is the CSE tuition fee per credit?" | Tuition Fees post | 1 | 1 |
  | "সিএসই ভর্তি ফি কত টাকা?" | Tuition Fees post | 1 | **none** |

  The Bangla query gets the right answer at rank 1 — from the vector branch
  alone. Every lexical rank is null.

### What generation does and does not guarantee

- **The answer language is chosen in code, not by the model.** Both local models
  tested ignored "reply in the question's language" — one answered Bangla
  questions in English, the other answered English questions in Bangla and
  rewrote `6,500` as `৬,৫০০`. `app/rag/language.py` decides, and the prompt is
  told.
- **Banglish is treated as Bangla.** A question written in Latin letters but in
  Bangla ("CSE er tuition fee koto") is answered in Bengali script, not English
  and not transliterated back. Script detection alone got this wrong, and
  `langdetect` scored 0/8 on it — labelling Banglish questions Dutch, Norwegian
  and Albanian — so detection is a function-word lexicon with a model fallback
  when the lexicon has no evidence.
- **Banglish retrieval is thinner than Bangla-script retrieval.** BGE-M3 ranked
  the right chunk first for every Banglish query probed, but the margin over the
  runner-up fell from +0.336 (English) to +0.061 for fully transliterated
  Banglish carrying no English loanwords. Measured on a 4-chunk toy corpus, so
  treat it as directional. Query normalisation is the follow-up if it bites.
- **A refusal is never model-generated.** When retrieval returns nothing, the
  chain returns a fixed sentence and never calls the model. Leaving that to a
  prompt rule made every refusal cite a `[1]` that did not exist.
- **Answer quality is not evaluated.** There is no scored eval set for
  generation; retrieval has one, generation does not.

## ⚠️ The seed data is synthetic

Every figure in `scripts/seed.py` — fees, deadlines, GPA thresholds, waivers —
is **invented**, marked `[SAMPLE DATA — NOT REAL DIU INFORMATION]`, and exists
only so the pipeline and tests could be built before real content was
available. Replace it with content copied from the official DIU admission site,
and replace the questions in `tests/test_retrieval_smoke.py` with questions
real applicants ask, before making any accuracy claim.

## Blocking dependency on the DB owner

`ai_content_vectors.vector` must be `VECTOR(1024)`, not `VECTOR(1536)` —
pgvector widths are exact, so BGE-M3's 1024-dim output is rejected outright by
a 1536 column. Full list in [`docs/schema-change-requests.md`](docs/schema-change-requests.md).

## Layout

```
app/
  main.py       routers + static mount, nothing else
  api/          HTTP only — validate, call a dependency, pick a status code
    admin.py        /api/health
    chat.py         /api/search (query side), /api/chat, /api/chat/stream
    documents.py    document lifecycle + ingest triggers
    dependencies.py what endpoints ask FastAPI for — the test override seam
  config/       settings — single source of truth
  database/     connection.py (psycopg3 pool) + repositories.py (all the SQL)
  embeddings/   BGE-M3 microservice + client
  ingestion/    chunker, pipeline, CLI entrypoint
  llm/          get_chat_model() — the one place ChatOllama is constructed
  rag/          cosine_search + bm25_search branches, hybrid_search (RRF),
                retriever wiring them together, langchain_retriever.py
                adapting it to LangChain, language.py (answer language +
                refusals), prompts.py, chain.py (the RAG flow itself)
  schemas/      Pydantic request/response models — the wire contract
  static/       dashboard (index.html), chatbot UI (chat.html)
migrations/     001_schema.sql
scripts/        setup_db.sh (database), seed.py (sample content)
tests/          unit tests run with nothing booted;
                anything needing Postgres or the embedding service is marked
                `integration` and skips when they are down
```

Two rules hold the layout together:

- **SQL lives in `database/repositories.py`, never in an endpoint.** Endpoints
  receive a repository through `Depends()`, so they can be tested against an
  in-memory one and no database running at all.
- **Ranking logic lives in `rag/hybrid_search.py` and takes no I/O.** Fusion is
  pure, so the rule that decides which chunk wins is tested with literals
  rather than by booting a database and reading the output.

```bash
uv run pytest            # unit tests only, ~0.3s, no infrastructure
uv run pytest -m integration   # needs the database + the embedding service
```
=======
🎓 **DIU Admission Chatbot (RAG-Powered)**

An intelligent, context-aware Admission Chatbot designed for Daffodil International University (DIU). This project leverages Retrieval-Augmented Generation (RAG), combining BM25 Keyword Search and Cosine Vector Similarity Search (Hybrid Search) to deliver highly accurate admission guidance in both English and Bengali.

**✨ Features**

Hybrid Search Retrieval: Combines BM25 lexical search with dense vector similarity search for precise document context extraction.

Multilingual Support: Handles inquiries seamlessly in English and Bengali.

FastAPI Backend: Lightweight, asynchronous, and high-performance REST APIs.

LangChain Integration: Advanced RAG pipeline for natural language generation and context management.

Ingestion Pipeline: Automatic text chunking, document parsing, and database seeding.

Admin & Chat Endpoints: Separate endpoints for client chat interaction and admin management.

**📁 Repository Structure**

diu_admission_chatbot/
├── app/
│   ├── api/          # API endpoints (chat, admin, documents, dependencies)
│   ├── config/       # Environment & app configurations
│   ├── database/     # DB connections & repository patterns
│   ├── embeddings/   # Vector embedding clients & services
│   ├── ingestion/    # Text chunker & ingestion pipeline
│   ├── llm/          # LLM integrations
│   ├── rag/          # Hybrid search, retriever, prompts, trace logic
│   ├── schemas/      # Pydantic data schemas
│   ├── static/       # HTML & Web UI interface
│   └── main.py       # FastAPI application entry point
├── docs/             # Documentation, user guides (BN), & architecture specs
├── migrations/       # Database SQL schema migrations
├── scripts/          # DB setup and data seeding scripts
├── .env.example      # Environment variables template
├── pyproject.toml    # Project dependencies
└── README.md         # Project documentation


🛠️** Tech Stack**

Backend Framework: FastAPI / Python

Orchestration / RAG: LangChain

Search & Retrieval: BM25 Search + Vector Cosine Search (Hybrid)

Database: PostgreSQL / Vector Store

Package Manager: uv / pip

🚀** Getting Started**

1. Prerequisites

Python 3.10+

PostgreSQL (with vector extension if applicable)

2. Installation & Setup

Clone the repository:

git clone https://github.com/your-username/diu-admission-chatbot.git
cd diu-admission-chatbot


Set up virtual environment & dependencies:

# Using uv
uv sync

# Or standard venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r pyproject.toml


Configure Environment Variables:
Copy .env.example to .env and fill in your API keys and DB credentials:

cp .env.example .env


Initialize Database & Seed Data:

bash scripts/setup_db.sh
python scripts/seed.py


Run the Application:

uvicorn app.main:app --reload


Access the Chatbot UI:
Open your browser and navigate to http://localhost:8000.

📄 **API Documentation**

Once the server is running, you can explore the interactive API docs:

Swagger UI: http://localhost:8000/docs

ReDoc: http://localhost:8000/redoc

📝** License**

This project is open-source and available under the MIT License.
>>>>>>> 5a3193ae54255eb17364107331b9fdf68edfb08d
