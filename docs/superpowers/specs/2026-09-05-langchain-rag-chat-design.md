# LangChain answer generation + chatbot UI

**Date:** 2026-09-05
**Status:** approved, not yet implemented
**Scope:** turn the retrieval-only slice into a grounded, streaming, bilingual chatbot.

## Why now, and what changes about the README's position

`README.md` currently lists "No answer generation yet" as a *deliberate* scope
limit, on the grounds that an LLM in front of unproven retrieval "would hide
retrieval problems behind fluent prose". That reasoning was correct and is now
satisfied, not discarded: `tests/test_retrieval_smoke.py` shows 10/10
question→post pairs in top-3 plus a Bangla query landing at rank 1. Retrieval is
measurably good, so generation may go on top of it.

Two guards keep the original concern alive after this change:

1. Sources are returned **alongside** the answer, taken from the retriever's own
   output — never parsed back out of generated text. A citation therefore cannot
   reference a chunk that was not retrieved.
2. The synthetic-corpus banner (§7) stays visible until the seed data is replaced.

## Decisions taken (and rejected alternatives)

| Decision | Chosen | Rejected, and why |
|---|---|---|
| LLM provider | **Ollama**, local | Claude/OpenAI would end the repo's "no API keys, no rate limits" property, which is its headline claim |
| LangChain depth | **Generation layer only** | `langchain-postgres` PGVector owns its own tables (`langchain_pg_collection`, `langchain_pg_embedding`) and cannot read the teammate-owned `ai_content_vectors`; adopting it means duplicating every vector or abandoning the shared schema |
| Chunking | **unchanged** | LangChain text splitters size by characters/tiktoken; Bangla expands to far more BGE-M3 subword tokens per character, so the existing tokenizer-based chunker is more correct here |
| Orchestration | **LCEL** | LangGraph adds a state machine for a fixed always-retrieve-then-answer flow; a ~3B-active local model is also unreliable at tool-calling |
| Memory | **client-held history** | Server-side sessions need new tables in a schema this repo does not own (see `docs/schema-change-requests.md`) |
| Frontend | **new vanilla `/chat` page** | React+Vite adds a Node toolchain, second dev server and build step to a Python-only repo; extending `index.html` mixes an applicant-facing bot into a staff ops console |

Nothing in `app/rag/{cosine_search,bm25_search,hybrid_search,retriever}.py`,
`app/database/`, `app/ingestion/` or `app/embeddings/` is modified. Every current
test stays green unchanged; that is a success criterion, not a hope.

## 1. Configuration

Appended to `app/config/settings.py` and `.env.example`:

```python
# --- Generation ----------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
# gemma3:4b. Chosen by probe, not by reputation: the previously-installed
# milkey/Kalomaze-Qwen3-16B-A3B cannot generate Bangla at all -- Ollama returns
# HTTP 500 ("output does not match the expected peg-native format") the moment a
# Bangla reply is requested. gemma3 is trained on 140+ languages and answers
# Bangla correctly. See §2.3.1 for the prompt work this still requires.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
# Set unconditionally, including on gemma3 which has no thinking mode (verified
# harmless there). It is what stops a swap to a Qwen3-family model from
# streaming raw chain-of-thought -- "Okay, the user is asking..." -- into the
# chat bubble as if it were the answer.
OLLAMA_DISABLE_THINKING = True
# Near-zero: this bot restates admission facts, it does not write prose.
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
# Chunks placed in the prompt. Lower than RETRIEVAL_TOP_K on purpose --
# retrieval casts wide, generation reads narrow.
GENERATION_TOP_K = int(os.getenv("GENERATION_TOP_K", "5"))
# Turns of history the client may send. Caps prompt growth.
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))
# The corpus is still scripts/seed.py. Flips the /chat warning banner off.
CORPUS_IS_SYNTHETIC = os.getenv("CORPUS_IS_SYNTHETIC", "true").lower() == "true"
```

## 2. Contracts (pinned — parallel work builds against these)

### 2.1 `app/llm/__init__.py`

```python
def get_chat_model(**overrides) -> BaseChatModel
```

Returns `ChatOllama` built from settings. Imported lazily inside the function so
importing `app.main` never imports `langchain_ollama`, matching the existing
lazy-import rule in `app/api/dependencies.py`.

### 2.2 `app/rag/langchain_retriever.py`

```python
class HybridRetriever(BaseRetriever):
    search_fn: Callable[..., Sequence[Hit]]   # defaults to app.rag.retriever.search
    top_k: int                                 # defaults to settings.GENERATION_TOP_K

    def _get_relevant_documents(self, query, *, run_manager) -> list[Document]
```

`Hit` → `Document` mapping, exhaustive and lossless:

| `Hit` field | `Document` |
|---|---|
| `chunk_text` | `page_content` |
| `ai_content_id`, `post_id`, `post_title`, `chunk_index`, `score`, `vector_rank`, `text_rank` | `metadata[...]`, same names |

`vector_rank`/`text_rank` must survive into metadata: they are the existing
debugging handle for "why did this chunk come back?", and losing them at the
LangChain boundary would remove it exactly where answers get harder to explain.

`search_fn` is a constructor field, so the adapter is unit-tested with a stub —
no Postgres, no embedding service, no Ollama.

### 2.3 `app/rag/prompts.py`

Module-level `ChatPromptTemplate` constants, so tests assert on the rules as
literals rather than on model behaviour:

- `CONTEXTUALIZE_PROMPT` — `(history, question) -> standalone question`. Rewrites
  "and for CSE?" into a self-contained query. Must instruct: return the question
  only, no answer, no preamble; preserve the original language.
- `ANSWER_PROMPT` — `(context, question, language_rule, refusal) -> answer`.
  Rules, each individually asserted by a test:
  1. Answer **only** from the numbered context. Never use outside knowledge.
  2. `{language_rule}` — the whole language instruction, injected per answer
     language from `LANGUAGE_RULES` (§2.3.2).
  3. Cite the sources used as `[1]`, `[2]`.
  4. If the context does not answer the question, reply with exactly `{refusal}`
     and nothing else — citing nothing.
  5. Never invent a number, date, fee or deadline.

  `{language_rule}` and `{refusal}` come from §2.3.1/§2.3.2, not from the model.
- `format_docs(docs) -> str` — numbered blocks, each headed by its `post_title`,
  so the model's `[n]` maps onto `sources[n-1]` positionally.

### 2.3.1 `app/rag/language.py` — a three-way language decision

**Script detection alone is wrong.** A large share of real DIU applicants write
*Banglish* — Bangla in Latin letters ("CSE er tuition fee koto", "amar admission
er jonno ki lagbe"). Script detection classifies that as English because the
characters are Latin, and the applicant gets an English answer to a Bangla
question.

Three input classes, two output languages:

| Question is | `QuestionLanguage` | Answer in |
|---|---|---|
| Bengali script | `bangla_script` | **Bangla** |
| Bangla in Latin letters | `banglish` | **Bangla** (Bengali script — never transliterated back) |
| Genuine English | `english` | English |

```python
QuestionLanguage = Literal["bangla_script", "banglish", "english"]
AnswerLanguage = Literal["Bangla", "English"]

ANSWER_LANGUAGE: dict[QuestionLanguage, AnswerLanguage]   # banglish -> "Bangla"
REFUSALS: dict[AnswerLanguage, str]

def classify_heuristic(text: str) -> QuestionLanguage | None: ...
```

`classify_heuristic` returns `None` for "not enough evidence", which is the
signal to fall back to the model. It is not an error case.

#### Why a hybrid, and not one or the other

Measured during design, on an 18-case set and then a 16-case adversarial set:

| Detector | Banglish | Adversarial | Cost |
|---|---|---|---|
| `langdetect` | **0/8** | — | — |
| Heuristic lexicon | 8/8 | 16/16 | 0 ms |
| `gemma3:4b` classifier | 8/8 | 15/16 | 0.33 s |

`langdetect` is not merely imperfect at Banglish, it is unusable: it labelled
the eight Banglish questions Dutch, Norwegian, Slovenian, Albanian, Afrikaans
and French. General language-ID is script-oriented, and romanized Bangla is
outside its training distribution. No off-the-shelf library is used.

The heuristic's 16/16 is **not** trustworthy on its own: the lexicon and the
test cases were written together, so it is scored on its own exam. Its real
failure mode is structural — Banglish built from vocabulary the lexicon lacks
falls through to `english`, which is precisely the requirement being violated.

The model's single miss (`"vorti procedure ta bolen"` → `bangla_script` rather
than `banglish`) is harmless: both map to a Bangla answer, so the decision that
matters was correct 16/16.

Hence: **heuristic decides when it has evidence, the model decides when it does
not.** The heuristic covers the common cases at zero cost; the model covers the
vocabulary the lexicon has never seen.

#### The latency is hidden, not paid

The fallback costs ~0.33 s, but nothing in retrieval depends on the answer
language. Classification and retrieval are therefore awaited together
(`asyncio.gather`), so the classifier finishes inside the time retrieval was
already taking and adds nothing to the critical path.

### 2.3.2 `LANGUAGE_RULES` — telling the model is not enough

Naming the language in the prompt does **not** make gemma3:4b answer a Banglish
question in Bangla. Probed with `language="Bangla"` and a plain instruction, it
answered 2 of 3 Banglish questions in English, and wrongly refused the third
although the context held the answer. The model anchors on the *question's*
script and overrides the instruction.

Two additions fixed it, and both are load-bearing:

1. Name the **script**, not the language — "Bengali script (বাংলা)", plus an
   explicit "even if the question is written with English letters".
2. A **one-shot example** of exactly that transformation.

Instruction alone scored 3/4 and leaked a stray Tamil character plus one
Banglish passthrough. Instruction + one-shot scored **4/4**, and the final
prompt scored **8/8** across the full matrix of
`{english, bangla_script, banglish}` × `{answerable, unanswerable}`.

`LANGUAGE_RULES: dict[AnswerLanguage, str]` therefore holds the *entire* rule-2
block per language, one-shot example included — not a language name that gets
interpolated into a shared sentence.

### 2.4 `app/rag/chain.py`

```python
async def astream_chat(question, history=(), *, retriever=None, llm=None
                       ) -> AsyncIterator[tuple[str, Any]]
async def achat(question, history=(), *, retriever=None, llm=None) -> dict
```

`astream_chat` yields `("sources", list[Document])` exactly once and first, then
zero or more `("token", str)`. `achat` returns `{"answer": str, "sources":
list[Document]}` and is implemented by consuming `astream_chat`, so the
streaming and non-streaming endpoints cannot drift into answering differently.

History arrives as plain `(role, content)` tuples, not `ChatMessage`; the
endpoint converts. `app/rag/` must not import `app/schemas/`, or the wire
contract and the retrieval core stop being separately testable.

Flow:

```
1. standalone = question, or CONTEXTUALIZE_PROMPT | llm | parser  (only if history)
2. concurrently (asyncio.gather):
     docs = retriever.ainvoke(standalone)
     kind = classify_heuristic(question) or <model classifier>
3. yield ("sources", docs)
4. if not docs:  yield ("token", REFUSALS[answer_language]);  return   # no model call
5. ANSWER_PROMPT | llm | parser, streamed as ("token", ...)
```

Step 2 is a `gather` rather than two awaits because retrieval does not depend on
the language and the language does not depend on retrieval. Running them
together hides the classifier's ~0.33 s inside time retrieval was already
spending.

**Written as an async generator, not as a composed `Runnable`.** Two reasons:
the "never call the model with no documents" rule is an early return here but
awkward inside a chain of `.assign()` calls, and streaming reads straight off
`.astream()` rather than being recovered by parsing `astream_events` output. The
pieces are still LCEL — `prompt | llm | parser`.

Contextualization is **skipped entirely** when `history` is empty: it is a whole
extra model round-trip, and a first question has nothing to be rewritten
against. Sources are yielded before any token and are never parsed back out of
the answer, which is what makes a fabricated citation structurally impossible.

Language is detected from the **user's original question**, never from the
rewritten standalone query — the rewrite is model output and may have drifted
into another language.

### 2.5 `app/schemas/chat.py` (additive — existing models untouched)

```python
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    history: list[ChatMessage] = Field(
        default_factory=list, max_length=MAX_HISTORY_TURNS * 2
    )
    top_k: int = Field(default=GENERATION_TOP_K, ge=1, le=20)

class ChatResponse(BaseModel):
    answer: str
    sources: list[SearchHit]      # the existing model, reused verbatim
```

`SearchHit` is reused rather than a parallel `Source` model: the fields are
identical, and two shapes for one concept would drift the moment either endpoint
gains a field. A `Document` from the retriever converts to a `SearchHit` by
reading `page_content` plus the metadata keys listed in §2.2.

### 2.6 `app/api/chat.py` (additive — `/api/search` unchanged)

`get_chain` joins `get_search` in `app/api/dependencies.py`, same lazy-import and
same `dependency_overrides` seam.

- `POST /api/chat` → `ChatResponse`. Non-streaming; the endpoint the test suite
  asserts against, because SSE assertions are brittle and this shares all the
  logic.
- `POST /api/chat/stream` → `text/event-stream`, `StreamingResponse`.

SSE frames, in this order, one JSON object per `data:` line:

```
{"type": "sources", "sources": [ ...SearchHit ]}  exactly one, first
{"type": "token",   "text": "..."}                zero or more
{"type": "done"}                                  exactly one, last
{"type": "error",   "detail": "..."}              terminal, replaces done
```

Sources arrive **first**, so the UI can render citations while tokens are still
streaming, and so a mid-stream failure still leaves the user with the retrieved
evidence rather than a blank panel.

## 3. `GET /chat` and the UI

`app/main.py` gains one route returning `app/static/chat.html`. `index.html`,
the staff ingestion dashboard, is not touched.

`chat.html` requirements:

- Reuses the CSS custom properties and `prefers-color-scheme` dark mode already
  defined in `index.html`. No framework, no build step, no CDN.
- Transcript of user/assistant bubbles; assistant text appended token-by-token.
- Citation chips under each answer, labelled with `post_title`, expanding on
  click to reveal the exact `chunk_text` — the applicant-facing equivalent of the
  `vector_rank`/`text_rank` debugging handle.
- Client keeps the history array, trimmed to the last `MAX_HISTORY_TURNS` pairs
  before each send.
- Empty state offering one English and one Bangla example question.
- Bangla renders correctly (font stack + `lang` handling).
- Reachability: input disabled with a clear message when `GET /api/health`
  reports a dependency down, rather than failing at submit time.

## 4. Grounding failure has two paths, and the common one is not the empty one

**Corrected during design probing.** The original text treated "retrieval
returned nothing" as *the* unanswerable case. It is the rare one. Hybrid search
returns the nearest `k` chunks however distant they are, so the dense branch
practically always returns something. Zero documents means an empty corpus, not
an unanswerable question.

The two paths:

1. **Zero documents** (empty corpus). The chain returns
   `REFUSALS[answer_language]` verbatim and **never calls the model**. Cheap
   insurance, and it keeps the empty-corpus case off the model entirely.
2. **Documents retrieved, none relevant** — the path real users hit. Here the
   model *does* see context, and rule 4 is what stops it inventing. Verified:
   asked about hostel fees against a context of tuition and deadline chunks, the
   model refused **5/5** across English, Banglish and Bangla-script questions,
   each time in the correct language and citing nothing.

Why this matters: given an empty context the same model invented "$360 per
year", "BDT 18,000" and "BDT 15,000", each with a fabricated `[1]`. Path 1 keeps
that input from ever reaching it; rule 4 plus real context is what holds on
path 2.

## 5. Testing (TDD, red → green per unit)

Follows the existing rule in `tests/conftest.py`: unit tests boot nothing;
anything needing infrastructure is `@pytest.mark.integration` and skips loudly.

| # | File | Asserts | Infra |
|---|---|---|---|
| 1 | `tests/rag/test_langchain_retriever.py` | every `Hit` field reaches `Document`; `vector_rank`/`text_rank` preserved incl. `None`; `top_k` forwarded | none |
| 2 | `tests/rag/test_language.py` | Bangla script → `bangla_script`; Banglish → `banglish`; English → `english`; no-evidence input → `None` (model fallback); `banglish` and `bangla_script` both map to the `Bangla` answer language; a `REFUSALS` and `LANGUAGE_RULES` entry exists per answer language | none |
| 3 | `tests/rag/test_prompts.py` | all five grounding rules present; `{language}`/`{refusal}` actually interpolated; `format_docs` numbering aligns `[n]` with `sources[n-1]`; titles included | none |
| 4 | `tests/rag/test_chain.py` | zero docs ⇒ the `REFUSALS` string for the question's language, **without** invoking the LLM; empty history ⇒ no contextualization call; non-empty ⇒ exactly one; `sources` pass through unmodified | `FakeListChatModel` |
| 5 | `tests/api/test_chat_endpoint.py` | `ChatResponse` shape; empty question ⇒ 422; over-long history ⇒ 422; SSE frame order `sources`→`token`*→`done` | fake chain via `dependency_overrides` |
| 6 | `tests/integration/test_generation.py` | English, Bangla-script **and Banglish** questions each produce a grounded, cited answer in the correct script; a question the retrieved chunks do not answer produces the exact refusal, in the right language | Ollama + PG + embeddings |

`conftest.py` gains an `ollama` fixture using the existing `_reachable` socket
probe against `OLLAMA_BASE_URL`, skipping with the command to start it.

**Sandbox note:** agent Bash tooling blocks `127.0.0.1:11434`, so tier 6 skips
inside the sandbox and must be run with the sandbox disabled. Tiers 1–5 are the
ones that gate implementation and need nothing.

## 6. Dependencies

Added to `pyproject.toml`: `langchain-core>=1.0`, `langchain-ollama>=1.0`.
Resolved during design probing to `langchain-core 1.6.2` / `langchain-ollama
1.1.0`. The floors say v1 deliberately: LangChain v1 is what the chain in §2.4
was verified against, and a `>=0.3` floor would silently permit the 0.x API.
Dev group: `pytest-asyncio>=0.24` for the streaming tests.

Not added: the `langchain` meta-package, `langchain-community`,
`langchain-postgres`. Each would pull a large transitive tree for code this
design does not use.

## 7. Synthetic-corpus banner

While `CORPUS_IS_SYNTHETIC` is true, `/chat` shows a persistent, non-dismissable
banner stating the answers come from sample data and are not real DIU
information. `README.md` already warns that every fee, deadline and GPA
threshold in `scripts/seed.py` is invented. A retrieved chunk is visibly a
chunk; a chatbot stating "the CSE tuition fee is BDT 6,500 per credit" in fluent
prose reads as fact. The banner is the difference between a demo and something
that misinforms an applicant.

## 8. Build order and parallelism

Sequential, because the contracts must exist before anything can build on them:

1. Config + dependencies (§1, §6)
2. Contracts as failing tests — tiers 1–3 (§5)

Then three streams run in parallel against the pinned contracts in §2:

- **A — chain:** `app/llm/`, `langchain_retriever.py`, `prompts.py`, `chain.py`
- **B — API:** `schemas/chat.py`, `api/dependencies.py`, `api/chat.py`, `main.py`
- **C — UI:** `app/static/chat.html`

Then, requiring all three:

3. Tier 4 endpoint tests, tier 5 integration tests
4. README update: move "No answer generation yet" out of *Deliberate scope
   limits*, document `/chat`, the Ollama prerequisite and the new settings

## 8.1 Known limits and tradeoffs

Accepted deliberately, and each measurable:

- **The heuristic lexicon is finite.** Banglish written with vocabulary it does
  not carry scores no evidence, which routes to the model classifier rather
  than to a wrong answer — but a *partial* match (one known Banglish word
  against two English stopwords) resolves to `english` without ever consulting
  the model. Growing the lexicon is the mitigation; there is no eval set for it.
- **The heuristic's reported accuracy is soft.** Lexicon and test cases were
  written together. Treat 16/16 as "works on cases it has words for", not as a
  generalisation estimate.
- **Banglish retrieval is thinner than Bangla-script retrieval.** BGE-M3 ranked
  the correct chunk first for every Banglish query probed, but the margin over
  the runner-up fell from +0.336 (English) to +0.061 for fully transliterated
  Banglish with no English loanwords. Banglish survives largely *because* it
  keeps English terms ("tuition", "credit", "CSE"). Measured against a 4-chunk
  toy corpus — directional, not conclusive. On a real corpus a +0.061 margin can
  flip. Not mitigated in this phase; a Banglish→Bangla query normalisation step
  is the obvious follow-up if it proves to be a problem.
- **The lexical branch contributes nothing for Banglish**, exactly as it already
  contributes nothing for Bangla script: `to_tsvector('english', ...)` cannot
  usefully tokenize either. Banglish retrieval is effectively vector-only.
- **The classifier adds a model round-trip on the fallback path only.** Hidden
  inside retrieval by `gather`, so the wall-clock cost is ~0 — but it does mean
  a question can consume two model calls before the answer, three with history.
- **Answer-language correctness is unmeasured at scale.** 8/8 on the design
  matrix and 5/5 on refusals is a smoke test, not an eval.

## 9. Out of scope

Reranking; the BGE-M3 sparse-weights fix for Bangla lexical search (still the
right fix, still a later phase); attachment/PDF ingestion; server-side
conversation persistence; authentication on `/chat`; answer-quality evaluation
harnesses; streaming the contextualization step.
