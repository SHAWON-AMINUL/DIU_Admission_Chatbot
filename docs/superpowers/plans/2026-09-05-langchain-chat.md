# LangChain Chat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the retrieval-only slice into a grounded, streaming, bilingual chatbot at `/chat`, powered by a local Ollama model through LangChain.

**Architecture:** LangChain sits at the generation layer only. The existing hybrid retriever (`app/rag/retriever.search`) is *wrapped* as a `BaseRetriever`, never replaced — `cosine_search`, `bm25_search`, RRF fusion, psycopg3 access and the BGE-M3 chunker are not modified. One async generator (`astream_chat`) is the single implementation of the RAG flow; the non-streaming entry point consumes it, so the two endpoints cannot drift.

**Tech Stack:** Python 3.13, FastAPI, `langchain-core` 1.x, `langchain-ollama` 1.x, Ollama (`gemma3:4b`), pytest, vanilla HTML/CSS/JS.

**Spec:** `docs/superpowers/specs/2026-09-05-langchain-rag-chat-design.md`

## Global Constraints

- **Do not modify** `app/rag/{retriever,cosine_search,bm25_search,hybrid_search,rows}.py`, `app/database/`, `app/ingestion/`, `app/embeddings/`, or `app/static/index.html`. Every existing test must still pass, unchanged, at every commit.
- `langchain-core>=1.0`, `langchain-ollama>=1.0`. Do **not** add `langchain`, `langchain-community`, or `langchain-postgres`.
- `ChatOllama` is always constructed with `reasoning=False`. Without it a Qwen3-family model streams raw chain-of-thought into the answer.
- Answer language is decided before the model answers, never inferred by it: `classify_heuristic` first, a model classifier only when the lexicon has no evidence. **Banglish (Bangla in Latin letters) answers in Bengali script**, never in English and never transliterated back. Refusals come from `REFUSALS`, never generated.
- Do not add a language-detection library. `langdetect` was measured at **0/8** on Banglish.
- `app/rag/` must never import from `app/schemas/`. The endpoint converts between them.
- Unit tests boot nothing. Anything needing Postgres, the embedding service or Ollama is `@pytest.mark.integration`.
- Follow the existing test style: module docstring explaining *why* the test exists, expected values worked by hand rather than read back from the implementation.
- Commit after every task with the message given in the task's final step.

## File Structure

| File | Responsibility |
|---|---|
| `app/rag/language.py` | *Create.* Script detection + per-language refusal strings. Pure. |
| `app/rag/langchain_retriever.py` | *Create.* `Hit` → `Document` adapter over the existing search. |
| `app/rag/prompts.py` | *Create.* Contextualize + answer templates, `format_docs`, `format_history`. |
| `app/llm/__init__.py` | *Create.* `get_chat_model()` → configured `ChatOllama`. |
| `app/rag/chain.py` | *Create.* `astream_chat` (the flow) + `achat` (consumes it). |
| `app/schemas/chat.py` | *Modify.* Add `ChatMessage`, `ChatRequest`, `ChatResponse`. |
| `app/api/dependencies.py` | *Modify.* Add `get_chat_fn` / `get_stream_fn` override seams. |
| `app/api/chat.py` | *Modify.* Add `POST /api/chat` and `POST /api/chat/stream`. |
| `app/main.py` | *Modify.* Add `GET /chat`. |
| `app/static/chat.html` | *Create.* The chatbot UI. |
| `app/config/settings.py` | *Modify.* Generation settings. |
| `tests/conftest.py` | *Modify.* Add the `ollama` fixture. |

## Parallelism

Tasks 1–2 are prerequisites for everything. After Task 6, three streams may run concurrently against the pinned interfaces: **A** = Task 7+8 (API), **B** = Task 9 (UI), **C** = Task 10 (integration). Tasks 3, 4, 5 may also run concurrently with each other once Task 2 lands.

---

### Task 1: Dependencies and configuration

**Files:**
- Modify: `pyproject.toml`
- Modify: `app/config/settings.py`
- Modify: `.env.example`

**Interfaces:**
- Consumes: nothing.
- Produces: `settings.OLLAMA_BASE_URL: str`, `settings.OLLAMA_MODEL: str`, `settings.OLLAMA_TEMPERATURE: float`, `settings.GENERATION_TOP_K: int`, `settings.MAX_HISTORY_TURNS: int`, `settings.CORPUS_IS_SYNTHETIC: bool`, `settings.OLLAMA_DISABLE_THINKING: bool`.

- [ ] **Step 1: Add the dependencies**

In `pyproject.toml`, add to the `dependencies` list:

```toml
    "langchain-core>=1.0",
    "langchain-ollama>=1.0",
```

And to the `dev` dependency group:

```toml
dev = ["pytest>=8.3", "pytest-asyncio>=0.24"]
```

- [ ] **Step 2: Configure pytest for async tests**

In `pyproject.toml`, inside `[tool.pytest.ini_options]`, add:

```toml
asyncio_mode = "auto"
```

- [ ] **Step 3: Install**

Run: `uv sync`
Expected: resolves and installs `langchain-core`, `langchain-ollama`, `pytest-asyncio`.

- [ ] **Step 4: Append generation settings**

Append to `app/config/settings.py`:

```python
# --- Generation -----------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# gemma3:4b, chosen by probe rather than reputation. The previously-installed
# milkey/Kalomaze-Qwen3-16B-A3B cannot generate Bangla at all -- Ollama returns
# HTTP 500 the moment a Bangla reply is requested. gemma3 is trained on 140+
# languages and answers Bangla correctly.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")

# Near-zero: this bot restates admission facts, it does not write prose.
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))

# Set unconditionally, including on gemma3 which has no thinking mode (verified
# harmless there). It is what stops a swap to a Qwen3-family model from
# streaming raw chain-of-thought into the chat bubble as if it were the answer.
OLLAMA_DISABLE_THINKING = True

# Chunks placed in the prompt. Lower than RETRIEVAL_TOP_K on purpose --
# retrieval casts wide, generation reads narrow.
GENERATION_TOP_K = int(os.getenv("GENERATION_TOP_K", "5"))

# Turns of history the client may send. Caps prompt growth.
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))

# The corpus is still scripts/seed.py, whose every figure is invented.
# Drives the warning banner on /chat.
CORPUS_IS_SYNTHETIC = os.getenv("CORPUS_IS_SYNTHETIC", "true").lower() == "true"
```

- [ ] **Step 5: Mirror into `.env.example`**

Append to `.env.example`:

```
# Generation (local Ollama; `ollama pull gemma3:4b`)
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=gemma3:4b
OLLAMA_TEMPERATURE=0.1
GENERATION_TOP_K=5
MAX_HISTORY_TURNS=6
# Set false once scripts/seed.py is replaced with real DIU content.
CORPUS_IS_SYNTHETIC=true
```

- [ ] **Step 6: Verify nothing regressed**

Run: `uv run pytest -q`
Expected: all existing tests pass, same count as before.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock app/config/settings.py .env.example
git commit -m "feat: add LangChain and Ollama generation settings"
```

---

### Task 2: Three-way language classification

**Files:**
- Create: `app/rag/language.py`
- Test: `tests/rag/test_language.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `QuestionLanguage = Literal["bangla_script", "banglish", "english"]`;
  `AnswerLanguage = Literal["Bangla", "English"]`;
  `ANSWER_LANGUAGE: dict[QuestionLanguage, AnswerLanguage]`;
  `REFUSALS: dict[AnswerLanguage, str]`;
  `classify_heuristic(text: str) -> QuestionLanguage | None`;
  `BANGLISH_MARKERS`, `ENGLISH_MARKERS`.

**Why this task exists:** a large share of DIU applicants write *Banglish* —
Bangla in Latin letters ("CSE er tuition fee koto"). Script detection calls that
English and answers in English, which is the requirement violated. Three input
classes collapse to two answer languages, and **both Bangla forms must answer in
Bengali script**.

`langdetect` was measured and rejected: it scored **0/8** on Banglish, labelling
those questions Dutch, Norwegian, Slovenian, Albanian, Afrikaans and French.
General language-ID is script-oriented; romanized Bangla is outside its training
distribution. Do not add a language-detection dependency.

`classify_heuristic` returns `None` for "not enough evidence". That is a normal
result, not an error — Task 6 falls back to a model call. Guessing `english`
there is precisely the failure being designed against.

- [ ] **Step 1: Write the failing test**

Create `tests/rag/test_language.py`:

```python
"""Which language the bot answers in, decided before the model sees anything.

Script detection alone is wrong here. A large share of real applicants write
Bangla in Latin letters -- "CSE er tuition fee koto" -- and a script check calls
that English. The rule this file pins is that both Bangla forms, script and
romanized, answer in Bengali script; only genuine English answers in English.
"""

import pytest

from app.rag.language import ANSWER_LANGUAGE, REFUSALS, classify_heuristic


@pytest.mark.parametrize(
    "text",
    ["সিএসই ভর্তি ফি কত টাকা?", "ভর্তি কবে শুরু হবে?"],
)
def test_bengali_script_is_recognised(text):
    assert classify_heuristic(text) == "bangla_script"


@pytest.mark.parametrize(
    "text",
    [
        "CSE er tuition fee koto",
        "amar admission er jonno ki lagbe",
        "CSE te vorti hote koto taka lagbe",
        "admission kobe shuru hobe",
        "hostel fee koto",
        "ami CSE te porte chai",
        "vorti procedure ta bolen",
        "result kharap hole ki hobe",
        "campus kothay",
        "ki ki lagbe",
        "admission test ache?",
        "DIU te CSE porte chaile result koto lagbe",
        "apply korar last date kobe",
    ],
)
def test_banglish_is_recognised_despite_latin_script(text):
    """The case a script check gets wrong, and the reason this module exists."""
    assert classify_heuristic(text) == "banglish"


@pytest.mark.parametrize(
    "text",
    [
        "What is the CSE tuition fee per credit?",
        "How much is the admission fee?",
        "Is there a scholarship for CSE?",
        "Where is the campus located?",
        "What are the requirements?",
        "I need info on the CSE program",
        "How much taka do I need?",
        "My back ache is bad",
        "Can I apply now?",
        "Is the fee per credit or per semester?",
    ],
)
def test_genuine_english_is_recognised(text):
    """'taka' and 'ache' are Banglish markers that also occur in English.

    Detection counts markers on both sides rather than matching any one, so a
    single collision cannot flip an English sentence.
    """
    assert classify_heuristic(text) == "english"


def test_a_mixed_script_question_counts_as_bangla_script():
    """'CSE ভর্তি ফি কত?' is a Bangla question with a borrowed acronym."""
    assert classify_heuristic("CSE ভর্তি ফি কত?") == "bangla_script"


@pytest.mark.parametrize(
    "text", ["", "DIU???", "Admission deadline?", "Scholarship criteria"]
)
def test_no_evidence_returns_none_rather_than_guessing(text):
    """None routes the question to the model classifier in Task 6.

    Defaulting to English here would silently reintroduce the whole bug:
    Banglish written with vocabulary the lexicon lacks would get an English
    answer, and nothing would report that it had happened.
    """
    assert classify_heuristic(text) is None


def test_both_bangla_forms_answer_in_bangla():
    """The hard requirement, stated once: Banglish in, Bengali script out."""
    assert ANSWER_LANGUAGE["bangla_script"] == "Bangla"
    assert ANSWER_LANGUAGE["banglish"] == "Bangla"
    assert ANSWER_LANGUAGE["english"] == "English"


def test_every_answer_language_has_a_refusal():
    """The no-documents path returns REFUSALS[...] verbatim; a missing key
    would be a KeyError on the one path that must never invent an answer."""
    assert set(REFUSALS) == {"English", "Bangla"}
    assert all(text.strip() for text in REFUSALS.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rag/test_language.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.rag.language'`

- [ ] **Step 3: Write the implementation**

Create `app/rag/language.py`:

```python
"""What language to answer in, and what to say when there is no answer.

Decided here rather than by the model, because the model gets it wrong. Probing
during design: gemma3:4b answered English questions in Bangla when left to
infer, and Qwen3 answered Bangla questions by echoing the English context.

Script detection alone is also wrong. Banglish -- Bangla written in Latin
letters, "CSE er tuition fee koto" -- is how a large share of applicants type,
and a script check calls it English. langdetect was measured on this and scored
0/8, labelling Banglish questions Dutch, Norwegian, Slovenian, Albanian,
Afrikaans and French. General language-ID is script-oriented and romanized
Bangla is outside its distribution, so there is no library to reach for.
"""

import re
from typing import Literal

QuestionLanguage = Literal["bangla_script", "banglish", "english"]
AnswerLanguage = Literal["Bangla", "English"]

# The Bengali Unicode block. Bangla uses no other range, and nothing else in
# scope uses this one, so a single character in it settles the script question.
_BENGALI_START = "ঀ"
_BENGALI_END = "৿"

# The requirement in one table: Banglish is Bangla, and answers in Bengali
# script -- never transliterated back into Latin letters.
ANSWER_LANGUAGE: dict[QuestionLanguage, AnswerLanguage] = {
    "bangla_script": "Bangla",
    "banglish": "Bangla",
    "english": "English",
}

# Returned verbatim when retrieval finds nothing, so this text must never
# depend on a model call. That is the whole point of the no-documents path.
REFUSALS: dict[AnswerLanguage, str] = {
    "English": (
        "I do not have that information. Please contact the admission office."
    ),
    "Bangla": (
        "এই তথ্য আমার কাছে নেই। অনুগ্রহ করে ভর্তি অফিসের সাথে যোগাযোগ করুন।"
    ),
}

# High-frequency Banglish function words. Function words, not nouns, because
# Banglish borrows English nouns freely -- "CSE er tuition fee koto" is mostly
# English vocabulary held together by Bangla grammar, and the grammar is the
# part that identifies it.
BANGLISH_MARKERS = frozenset("""
er te ta ti ki ke koto kto kobe kokhon kothay kivabe kemon kmn ache achhe asche
hobe hoy hoye lagbe lage korte kore korbo parbo dite nite jonno jonne amar ami
amake apnar apni tumi tar nai nei bhorti vorti taka kichu kisu kono shob sob
bhalo valo chai chay jodi tahole abar onek aro kina porte pora shuru suru ekhon
dorkar holo ase hoise hole kharap bolen korar chaile
""".split())

# Counted against the above rather than used alone: some Banglish markers are
# also English words ("ache", "taka", "hole"), so one collision must not be
# able to flip an English sentence.
ENGLISH_MARKERS = frozenset("""
the is are was what how much many do does did i you we my your our a an of for
to in on at can could when where which who why need want tell me about there
offer with per and or if from have has get give it this that any all
""".split())


def classify_heuristic(text: str) -> QuestionLanguage | None:
    """Classify by script, then by function-word evidence.

    Returns None when neither vocabulary appears. That is deliberately not
    "english": a question the lexicon has no words for is exactly the case
    where a wrong guess reintroduces the bug, so the caller asks the model.
    """
    if any(_BENGALI_START <= char <= _BENGALI_END for char in text):
        return "bangla_script"

    words = re.findall(r"[a-z]+", text.lower())
    banglish = sum(word in BANGLISH_MARKERS for word in words)
    english = sum(word in ENGLISH_MARKERS for word in words)

    if banglish > english:
        return "banglish"
    if english > banglish:
        return "english"
    return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/rag/test_language.py -v`
Expected: 32 passed.

- [ ] **Step 5: Commit**

```bash
git add app/rag/language.py tests/rag/test_language.py
git commit -m "feat: classify question language three ways, including Banglish"
```

---

### Task 3: LangChain retriever adapter

**Files:**
- Create: `app/rag/langchain_retriever.py`
- Test: `tests/rag/test_langchain_retriever.py`

**Interfaces:**
- Consumes: `app.rag.retriever.Hit` (existing, unmodified), `settings.GENERATION_TOP_K`.
- Produces: `HybridRetriever(search_fn=..., top_k=...)`, a `langchain_core.retrievers.BaseRetriever` whose `.invoke(query)` returns `list[Document]`.

- [ ] **Step 1: Write the failing test**

Create `tests/rag/test_langchain_retriever.py`:

```python
"""The seam between the existing hybrid retriever and LangChain.

The adapter is the only place a Hit becomes a Document. If a field is dropped
here it is gone from every answer and every citation downstream, so this test
pins the whole mapping rather than sampling it.
"""

import pytest

from app.rag.langchain_retriever import HybridRetriever
from app.rag.retriever import Hit

HIT = Hit(
    ai_content_id=42,
    post_id=7,
    post_title="CSE Tuition Fees",
    chunk_index=0,
    chunk_text="The fee is BDT 6500 per credit.",
    score=0.032258,
    vector_rank=2,
    text_rank=None,
)


@pytest.fixture
def calls():
    return []


@pytest.fixture
def retriever(calls):
    def fake_search(query: str, top_k: int):
        calls.append((query, top_k))
        return [HIT]

    return HybridRetriever(search_fn=fake_search, top_k=3)


def test_chunk_text_becomes_the_document_body(retriever):
    """page_content is what the model actually reads."""
    docs = retriever.invoke("CSE tuition")

    assert len(docs) == 1
    assert docs[0].page_content == "The fee is BDT 6500 per credit."


def test_every_hit_field_survives_into_metadata(retriever):
    """Citations are rendered from metadata, so a dropped field is a lost citation."""
    docs = retriever.invoke("CSE tuition")

    assert docs[0].metadata == {
        "ai_content_id": 42,
        "post_id": 7,
        "post_title": "CSE Tuition Fees",
        "chunk_index": 0,
        "score": 0.032258,
        "vector_rank": 2,
        "text_rank": None,
    }


def test_a_missing_branch_stays_none_rather_than_zero(retriever):
    """text_rank=None means 'the lexical branch never returned this chunk'.

    Collapsing that to 0 would read as 'ranked first', which is the opposite
    fact, and it is the handle used to debug a wrong answer.
    """
    docs = retriever.invoke("CSE tuition")

    assert docs[0].metadata["text_rank"] is None
    assert docs[0].metadata["vector_rank"] == 2


def test_top_k_is_forwarded_to_the_underlying_search(retriever, calls):
    """Generation reads fewer chunks than retrieval returns."""
    retriever.invoke("CSE tuition")

    assert calls == [("CSE tuition", 3)]


def test_no_hits_yields_no_documents(calls):
    """The empty case is a real path: it is what triggers the refusal."""
    retriever = HybridRetriever(search_fn=lambda query, top_k: [], top_k=5)

    assert retriever.invoke("hostel fees") == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rag/test_langchain_retriever.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.rag.langchain_retriever'`

- [ ] **Step 3: Write the implementation**

Create `app/rag/langchain_retriever.py`:

```python
"""Presents the existing hybrid retriever to LangChain as a BaseRetriever.

An adapter, deliberately: the dense branch, the lexical branch and the RRF
fusion are already tested and already work, and langchain-postgres' PGVector
could not read this schema anyway -- it owns its own tables, and the schema
here belongs to another teammate.

Nothing in app/rag/retriever.py changes. This module only reshapes the result.
"""

from collections.abc import Callable, Sequence

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from app.config import settings
from app.rag.retriever import Hit


def _default_search(query: str, top_k: int) -> Sequence[Hit]:
    """Imported lazily: importing this module must not require numpy,
    pgvector, or a reachable embedding service. It is what lets the tests
    above run with nothing booted."""
    from app.rag.retriever import search

    return search(query, top_k=top_k)


class HybridRetriever(BaseRetriever):
    """The project's hybrid search, wearing a LangChain interface."""

    # A field rather than a module-level import, so a test can substitute
    # retrieval wholesale without Postgres or the embedding service.
    search_fn: Callable[..., Sequence[Hit]] = _default_search
    top_k: int = Field(default_factory=lambda: settings.GENERATION_TOP_K)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return [
            Document(
                page_content=hit.chunk_text,
                # vector_rank/text_rank travel with the chunk on purpose: they
                # are the existing answer to "why did this chunk come back?",
                # and an answer is exactly where that question gets asked.
                metadata={
                    "ai_content_id": hit.ai_content_id,
                    "post_id": hit.post_id,
                    "post_title": hit.post_title,
                    "chunk_index": hit.chunk_index,
                    "score": hit.score,
                    "vector_rank": hit.vector_rank,
                    "text_rank": hit.text_rank,
                },
            )
            for hit in self.search_fn(query, top_k=self.top_k)
        ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/rag/test_langchain_retriever.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add app/rag/langchain_retriever.py tests/rag/test_langchain_retriever.py
git commit -m "feat: adapt hybrid retriever to the LangChain retriever interface"
```

---

### Task 4: Prompts and context formatting

**Files:**
- Create: `app/rag/prompts.py`
- Test: `tests/rag/test_prompts.py`

**Interfaces:**
- Consumes: `app.rag.language.AnswerLanguage`.
- Produces: `LANGUAGE_RULES: dict[AnswerLanguage, str]`;
  `ANSWER_PROMPT: ChatPromptTemplate` (variables `context`, `question`,
  `language_rule`, `refusal`); `CONTEXTUALIZE_PROMPT` (variables `history`,
  `question`); `CLASSIFY_PROMPT` (variable `question`);
  `format_docs(docs) -> str`; `format_history(history) -> str`.

**Why the Bangla rule is a whole block, not a language name:** naming the
language does not work. Probed with `language="Bangla"` and a plain instruction,
gemma3:4b answered 2 of 3 Banglish questions in **English** and wrongly refused
the third. It anchors on the question's script and overrides the instruction.
Two additions fixed it and both are load-bearing — naming the *script* ("Bengali
script (বাংলা)", plus "even if the question is written with English letters"),
and a **one-shot example** of that exact transformation. Instruction alone
scored 3/4 and leaked a stray Tamil character; instruction plus one-shot scored
4/4, and 8/8 across the full language × answerable matrix. Do not shorten this
rule.

- [ ] **Step 1: Write the failing test**

Create `tests/rag/test_prompts.py`:

```python
"""The grounding rules, asserted as text rather than as model behaviour.

What a local 4B model does with these rules is checked in the integration
tests. What is checked here is that the rules are in the prompt at all, and that
the numbering the model is told to cite matches the numbering the UI renders --
an off-by-one there silently attributes every fact to the wrong post.
"""

from langchain_core.documents import Document

from app.rag.prompts import (
    ANSWER_PROMPT,
    CLASSIFY_PROMPT,
    CONTEXTUALIZE_PROMPT,
    LANGUAGE_RULES,
    format_docs,
    format_history,
)

DOCS = [
    Document(page_content="Fee is BDT 6500 per credit.", metadata={"post_title": "Tuition Fees"}),
    Document(page_content="Applications close 30 June.", metadata={"post_title": "Deadlines"}),
]


def test_context_is_numbered_from_one_with_titles():
    """The model is told to cite [1]; the UI renders sources[0] for it."""
    formatted = format_docs(DOCS)

    assert "[1] Tuition Fees" in formatted
    assert "[2] Deadlines" in formatted
    assert "Fee is BDT 6500 per credit." in formatted


def test_empty_context_formats_without_raising():
    assert format_docs([]) == ""


def test_history_is_labelled_by_speaker():
    formatted = format_history([("user", "What is the fee?"), ("assistant", "BDT 6500.")])

    assert "What is the fee?" in formatted
    assert "BDT 6500." in formatted


def test_the_bangla_rule_names_the_script_and_covers_banglish():
    """Naming the language is not enough -- the model answered Banglish
    questions in English until the rule named the script and said explicitly
    that a Latin-script question still gets a Bengali-script answer."""
    rule = LANGUAGE_RULES["Bangla"]

    assert "Bengali script" in rule
    assert "English letters" in rule
    assert "0-9" in rule


def test_the_bangla_rule_carries_a_one_shot_example():
    """The example is load-bearing: instruction alone scored 3/4 on Banglish
    and leaked a non-Bengali character. With it, 4/4."""
    rule = LANGUAGE_RULES["Bangla"]

    assert "Example:" in rule
    # The example's answer must itself be in Bengali script, or it teaches
    # the opposite of the rule.
    answer_line = rule.split("Answer:")[-1]
    assert any("ঀ" <= char <= "৿" for char in answer_line)


def test_the_english_rule_does_not_mention_bengali():
    """The rule is injected wholesale, so leaking Bangla instructions into an
    English answer is a real failure mode."""
    assert "Bengali" not in LANGUAGE_RULES["English"]
    assert "English" in LANGUAGE_RULES["English"]


def test_the_answer_prompt_states_every_grounding_rule():
    """Each rule earns its place by a failure seen during design probing."""
    rendered = ANSWER_PROMPT.format(
        context="[1] Tuition Fees\nFee is BDT 6500 per credit.",
        question="What is the fee?",
        language_rule=LANGUAGE_RULES["English"],
        refusal="I do not have that information.",
    )

    lowered = rendered.lower()
    assert "only" in lowered                              # rule 1: context-only
    assert "english" in lowered                           # rule 2: injected rule
    assert "[1]" in rendered                              # rule 3: citation format
    assert "I do not have that information." in rendered  # rule 4: exact refusal
    assert "invent" in lowered                            # rule 5: no invented figures


def test_the_answer_prompt_carries_the_context_and_question():
    rendered = ANSWER_PROMPT.format(
        context="[1] Tuition Fees\nFee is BDT 6500 per credit.",
        question="What is the fee?",
        language_rule=LANGUAGE_RULES["Bangla"],
        refusal="refusal text",
    )

    assert "Fee is BDT 6500 per credit." in rendered
    assert "What is the fee?" in rendered
    assert "Bengali script" in rendered


def test_the_contextualize_prompt_asks_for_a_question_not_an_answer():
    """It runs before retrieval. If it answers instead of rewriting, the
    retrieval query becomes an invented answer and retrieval goes wrong."""
    rendered = CONTEXTUALIZE_PROMPT.format(
        history="user: What is the CSE fee?\nassistant: BDT 6500 per credit.",
        question="and for BBA?",
    )

    assert "do not answer" in rendered.lower()
    assert "and for BBA?" in rendered


def test_the_classify_prompt_offers_exactly_the_three_labels():
    """Its output is looked up in ANSWER_LANGUAGE, so a fourth label would
    KeyError on the fallback path."""
    rendered = CLASSIFY_PROMPT.format(question="CSE er fee koto")

    assert "bangla_script" in rendered
    assert "banglish" in rendered
    assert "english" in rendered
    assert "CSE er fee koto" in rendered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rag/test_prompts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.rag.prompts'`

- [ ] **Step 3: Write the implementation**

Create `app/rag/prompts.py`:

```python
"""The prompts, and the formatting they depend on.

Every rule below was added because a model probed during design broke it.
gemma3:4b answered English questions in Bangla when left to infer the language,
rewrote 6,500 as ৬,৫০০, answered Banglish questions in English when merely told
"answer in Bangla", and appended a bogus "[1]" to refusals -- citing a source
for information it had just said was absent.
"""

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.rag.language import AnswerLanguage

# The entire language instruction, per answer language, injected wholesale as
# rule 2. Not a language name interpolated into a shared sentence: naming the
# language was measured and does not work on Banglish input.
LANGUAGE_RULES: dict[AnswerLanguage, str] = {
    # Three things are load-bearing here and none may be trimmed: naming the
    # *script* rather than the language, saying explicitly that a Latin-script
    # question still gets a Bengali-script answer, and the one-shot example.
    # Instruction alone scored 3/4 on Banglish and leaked a Tamil character;
    # with the example, 4/4.
    "Bangla": """Write your ENTIRE answer in Bengali script (বাংলা). Never use \
Latin letters for Bangla words. Even if the question is written with English \
letters (Banglish), you MUST still reply in Bengali script. Use Western digits \
(0-9) for all numbers.

Example:
Question: CSE er fee koto
Answer: সিএসই টিউশন ফি প্রতি ক্রেডিট BDT 6,500 [1]।""",
    "English": (
        "Write your ENTIRE answer in English. Use Western digits (0-9) for all numbers."
    ),
}

_ANSWER_SYSTEM = """You are the Daffodil International University admission assistant.

Rules, in priority order:
1. Answer ONLY using the numbered context. Never use outside knowledge.
2. {language_rule}
3. End each sentence you took from the context with its citation, like [1].
4. If the context does not answer the question, reply with exactly this and \
nothing else: {refusal}
5. Never invent a number, date, fee or deadline."""

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _ANSWER_SYSTEM),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

# Runs before retrieval, so its output becomes the search query. If it answers
# rather than rewrites, retrieval searches for an invented answer.
_CONTEXTUALIZE_SYSTEM = """Rewrite the user's follow-up into a standalone question \
that can be understood without the conversation.

Return ONLY the rewritten question. Do not answer it. Do not explain.
Keep it in the same language and script the user wrote it in.
If it already stands alone, return it unchanged."""

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _CONTEXTUALIZE_SYSTEM),
        ("human", "Conversation so far:\n{history}\n\nFollow-up: {question}"),
    ]
)

# The fallback when the lexicon has no evidence either way. Labels match
# QuestionLanguage exactly, because the result is looked up in ANSWER_LANGUAGE.
_CLASSIFY_SYSTEM = """Classify the user's question into exactly one label:

bangla_script = written in Bengali letters
banglish = Bangla language written with English letters (e.g. "fee koto", "ki lagbe")
english = ordinary English

Reply with the label only."""

CLASSIFY_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _CLASSIFY_SYSTEM), ("human", "{question}")]
)


def format_docs(docs: Sequence[Document]) -> str:
    """Number the chunks from 1, headed by post title.

    The numbering is a contract between three places: what the model is told to
    cite, what this emits, and what the UI renders as sources[n-1].
    """
    return "\n\n".join(
        f"[{index}] {doc.metadata.get('post_title', 'Untitled')}\n{doc.page_content}"
        for index, doc in enumerate(docs, start=1)
    )


def format_history(history: Sequence[tuple[str, str]]) -> str:
    """(role, content) pairs as plain labelled lines."""
    return "\n".join(f"{role}: {content}" for role, content in history)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/rag/test_prompts.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add app/rag/prompts.py tests/rag/test_prompts.py
git commit -m "feat: add grounded prompts with per-language rule blocks"
```

---

### Task 5: The chat model factory

**Files:**
- Create: `app/llm/__init__.py`
- Test: `tests/test_llm_factory.py`

**Interfaces:**
- Consumes: `settings.OLLAMA_*`.
- Produces: `get_chat_model(**overrides) -> BaseChatModel`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_llm_factory.py`:

```python
"""How the chat model is constructed. No model is contacted.

reasoning=False is the load-bearing line. With Ollama's default, a Qwen3-family
model streams its raw chain of thought -- "Okay, the user is asking..." -- into
the answer field, and the chat UI renders it as the reply.
"""

from app.config import settings
from app.llm import get_chat_model


def test_the_model_is_built_from_settings():
    llm = get_chat_model()

    assert llm.model == settings.OLLAMA_MODEL
    assert llm.temperature == settings.OLLAMA_TEMPERATURE


def test_thinking_is_disabled():
    """Verified harmless on gemma3, which has no thinking mode, so it can be
    set unconditionally and keep OLLAMA_MODEL genuinely swappable."""
    assert get_chat_model().reasoning is False


def test_overrides_win():
    """The integration tests need a shorter num_predict than production."""
    assert get_chat_model(temperature=0.9).temperature == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_llm_factory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.llm'`

- [ ] **Step 3: Write the implementation**

Create `app/llm/__init__.py`:

```python
"""The chat model, built in one place.

Local Ollama, so the project keeps the property its README leads with: no API
keys, no rate limits, no cloud dependency. Embeddings were already local; this
keeps generation local too.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel

from app.config import settings


def get_chat_model(**overrides: Any) -> BaseChatModel:
    """A configured ChatOllama. Overrides win, for tests and integration runs.

    Imported lazily so that importing app.main does not pull langchain_ollama
    and httpx into every process that only wants to serve static files.
    """
    from langchain_ollama import ChatOllama

    params: dict[str, Any] = {
        "model": settings.OLLAMA_MODEL,
        "base_url": settings.OLLAMA_BASE_URL,
        "temperature": settings.OLLAMA_TEMPERATURE,
        # Not optional. Without it a Qwen3-family model emits its chain of
        # thought as the answer, and the UI has no way to tell the difference.
        "reasoning": not settings.OLLAMA_DISABLE_THINKING,
    }
    params.update(overrides)
    return ChatOllama(**params)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_llm_factory.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add app/llm/__init__.py tests/test_llm_factory.py
git commit -m "feat: add the Ollama chat model factory"
```

---

### Task 6: The RAG chain

**Files:**
- Create: `app/rag/chain.py`
- Test: `tests/rag/test_chain.py`

**Interfaces:**
- Consumes: `HybridRetriever`, `ANSWER_PROMPT`, `CONTEXTUALIZE_PROMPT`, `CLASSIFY_PROMPT`, `LANGUAGE_RULES`, `format_docs`, `format_history`, `classify_heuristic`, `ANSWER_LANGUAGE`, `REFUSALS`, `get_chat_model`.
- Produces:
  - `astream_chat(question: str, history: Sequence[tuple[str, str]], *, retriever=None, llm=None) -> AsyncIterator[tuple[str, Any]]` — yields `("sources", list[Document])` exactly once and first, then zero or more `("token", str)`.
  - `achat(question: str, history: Sequence[tuple[str, str]], *, retriever=None, llm=None) -> dict` — `{"answer": str, "sources": list[Document]}`.

**Design note:** `achat` is implemented by consuming `astream_chat`. One implementation, two entry points, so the streaming and non-streaming endpoints cannot answer differently.

- [ ] **Step 1: Write the failing test**

Create `tests/rag/test_chain.py`:

```python
"""The RAG flow, with retrieval and the model both faked.

Answer quality is an integration concern. What is pinned here is the control
flow: that a refusal never reaches the model, that a first question costs one
model call rather than two, and that the sources handed to the UI are the ones
retrieval actually returned.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.rag.chain import achat, astream_chat
from app.rag.language import REFUSALS
from app.rag.langchain_retriever import HybridRetriever
from app.rag.retriever import Hit

HIT = Hit(
    ai_content_id=42,
    post_id=7,
    post_title="CSE Tuition Fees",
    chunk_index=0,
    chunk_text="The fee is BDT 6500 per credit.",
    score=0.032258,
    vector_rank=1,
    text_rank=1,
)


@pytest.fixture
def llm():
    """FakeListChatModel counts its own invocations in `.i`.

    The list is longer than any test's call count on purpose: the counter wraps
    back to 0 once the responses are exhausted, so a two-element list would
    make a two-call test look like a zero-call test.
    """
    return FakeListChatModel(responses=["the answer"] * 5)


@pytest.fixture
def follow_up_llm():
    """First response is the rewritten query, second is the answer."""
    return FakeListChatModel(
        responses=["standalone question", "the answer", "unused", "unused", "unused"]
    )


@pytest.fixture
def retriever():
    return HybridRetriever(search_fn=lambda query, top_k: [HIT], top_k=5)


@pytest.fixture
def empty_retriever():
    return HybridRetriever(search_fn=lambda query, top_k: [], top_k=5)


async def test_an_answer_carries_the_documents_retrieval_returned(retriever, llm):
    """Sources travel beside the answer, never parsed back out of it.

    That is what makes a citation pointing at a chunk that was never retrieved
    structurally impossible rather than merely unlikely.
    """
    result = await achat("What is the CSE fee?", [], retriever=retriever, llm=llm)

    assert result["answer"] == "the answer"
    assert [doc.metadata["post_id"] for doc in result["sources"]] == [7]


async def test_no_documents_means_no_model_call_at_all(empty_retriever, llm):
    """The most important behaviour in the system must not depend on the model
    obeying a prompt rule. With nothing retrieved there is nothing to ground an
    answer in, so the refusal is returned directly."""
    result = await achat("What is the hostel fee?", [], retriever=empty_retriever, llm=llm)

    assert result["answer"] == REFUSALS["English"]
    assert result["sources"] == []
    assert llm.i == 0


async def test_a_bangla_question_refuses_in_bangla(empty_retriever, llm):
    result = await achat("হোস্টেল ফি কত?", [], retriever=empty_retriever, llm=llm)

    assert result["answer"] == REFUSALS["Bangla"]


async def test_a_first_question_costs_one_model_call(retriever, llm):
    """Contextualization is a whole extra round-trip. On a local model that is
    the difference between a snappy first question and a sluggish one, and with
    no history there is nothing to contextualize against."""
    await achat("What is the CSE fee?", [], retriever=retriever, llm=llm)

    assert llm.i == 1


async def test_a_follow_up_is_rewritten_before_retrieval(follow_up_llm):
    """'and for BBA?' retrieves nothing useful on its own.

    The rewritten question, not the raw follow-up, is what reaches retrieval.
    """
    seen = []

    def spy_search(query, top_k):
        seen.append(query)
        return [HIT]

    retriever = HybridRetriever(search_fn=spy_search, top_k=5)
    await achat(
        "and for BBA?",
        [("user", "What is the CSE fee?"), ("assistant", "BDT 6500.")],
        retriever=retriever,
        llm=follow_up_llm,
    )

    assert follow_up_llm.i == 2
    assert seen == ["standalone question"]


async def test_streaming_emits_sources_first_then_tokens(retriever, llm):
    """The UI renders citations while tokens are still arriving, and a
    mid-stream failure still leaves the user with the evidence."""
    events = [event async for event in astream_chat("fee?", [], retriever=retriever, llm=llm)]

    assert events[0][0] == "sources"
    assert all(kind == "token" for kind, _ in events[1:])
    assert "".join(text for _, text in events[1:]) == "the answer"


async def test_a_banglish_question_refuses_in_bangla(empty_retriever, llm):
    """The hard requirement, on the refusal path: Banglish is Bangla.

    Script detection would call this English and answer in English.
    """
    result = await achat("hostel fee koto", [], retriever=empty_retriever, llm=llm)

    assert result["answer"] == REFUSALS["Bangla"]
    assert llm.i == 0


async def test_a_confident_heuristic_never_calls_the_classifier(retriever, llm):
    """One model call for the answer, none for classification.

    The lexicon covers the common cases for free; paying a round-trip for
    "CSE er tuition fee koto" would be pure waste.
    """
    await achat("CSE er tuition fee koto", [], retriever=retriever, llm=llm)

    assert llm.i == 1


async def test_an_unrecognised_question_falls_back_to_the_classifier(empty_retriever):
    """'Admission deadline?' carries no marker from either lexicon.

    Guessing English there is the bug this design exists to prevent, so the
    model is asked instead -- and its label decides the refusal language.
    """
    llm = FakeListChatModel(responses=["banglish", "unused", "unused"])

    result = await achat("Admission deadline?", [], retriever=empty_retriever, llm=llm)

    assert llm.i == 1
    assert result["answer"] == REFUSALS["Bangla"]


async def test_an_unparseable_classifier_reply_falls_back_to_english(empty_retriever):
    """A 4B model asked for one word will sometimes return a sentence.

    Defaulting to English is the safe direction: an English answer to a Bangla
    question is bad, but a KeyError is a 500.
    """
    llm = FakeListChatModel(responses=["I think it is probably a question", "x", "x"])

    result = await achat("Admission deadline?", [], retriever=empty_retriever, llm=llm)

    assert result["answer"] == REFUSALS["English"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/rag/test_chain.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.rag.chain'`

- [ ] **Step 3: Write the implementation**

Create `app/rag/chain.py`:

```python
"""The RAG flow: classify, contextualize, retrieve, answer.

Written as one async generator rather than a single composed Runnable. Two
reasons, both practical: the "never call the model with no documents" rule is
plainly visible as an early return but awkward inside a chain of .assign()
calls, and streaming reads straight off .astream() instead of being recovered
by parsing astream_events output.

The pieces are still LCEL -- prompt | llm | parser -- and achat() is
implemented by consuming astream_chat(), so the streaming and non-streaming
endpoints cannot drift into answering differently.
"""

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever

from app.llm import get_chat_model
from app.rag.langchain_retriever import HybridRetriever
from app.rag.language import (
    ANSWER_LANGUAGE,
    REFUSALS,
    QuestionLanguage,
    classify_heuristic,
)
from app.rag.prompts import (
    ANSWER_PROMPT,
    CLASSIFY_PROMPT,
    CONTEXTUALIZE_PROMPT,
    LANGUAGE_RULES,
    format_docs,
    format_history,
)

_LABELS: tuple[QuestionLanguage, ...] = ("bangla_script", "banglish", "english")


async def _classify(question: str, llm: BaseChatModel) -> QuestionLanguage:
    """Lexicon first, model only when the lexicon has no evidence.

    The heuristic is free and covers the common cases; the model covers the
    Banglish vocabulary the lexicon has never seen. Guessing English on no
    evidence would silently reintroduce the bug this whole path exists for.
    """
    detected = classify_heuristic(question)
    if detected is not None:
        return detected

    raw = await (CLASSIFY_PROMPT | llm | StrOutputParser()).ainvoke(
        {"question": question}
    )
    lowered = raw.strip().lower()
    for label in _LABELS:
        if label in lowered:
            return label
    # A small model asked for one word sometimes returns a sentence. English is
    # the safe default: a wrong language is bad, an unhandled KeyError is a 500.
    return "english"


async def astream_chat(
    question: str,
    history: Sequence[tuple[str, str]] = (),
    *,
    retriever: BaseRetriever | None = None,
    llm: BaseChatModel | None = None,
) -> AsyncIterator[tuple[str, Any]]:
    """Yield ("sources", docs) once, then ("token", text) repeatedly."""
    retriever = retriever if retriever is not None else HybridRetriever()
    llm = llm if llm is not None else get_chat_model()

    # Skipped entirely without history: it is a second model round-trip, and a
    # first question has nothing to be rewritten against.
    standalone = question
    if history:
        contextualize = CONTEXTUALIZE_PROMPT | llm | StrOutputParser()
        standalone = (
            await contextualize.ainvoke(
                {"history": format_history(history), "question": question}
            )
        ).strip() or question

    # Retrieval does not need the language and the language does not need
    # retrieval, so the classifier's round-trip hides inside time retrieval was
    # already spending rather than adding to it.
    #
    # The language is read from the user's own question, never from `standalone`
    # -- that is model output and may have drifted into another language.
    docs, kind = await asyncio.gather(
        retriever.ainvoke(standalone),
        _classify(question, llm),
    )
    yield "sources", docs

    language = ANSWER_LANGUAGE[kind]

    if not docs:
        yield "token", REFUSALS[language]
        return

    answer = ANSWER_PROMPT | llm | StrOutputParser()
    async for token in answer.astream(
        {
            "context": format_docs(docs),
            "question": standalone,
            "language_rule": LANGUAGE_RULES[language],
            "refusal": REFUSALS[language],
        }
    ):
        if token:
            yield "token", token


async def achat(
    question: str,
    history: Sequence[tuple[str, str]] = (),
    *,
    retriever: BaseRetriever | None = None,
    llm: BaseChatModel | None = None,
) -> dict[str, Any]:
    """The whole answer at once. Consumes astream_chat so there is one flow."""
    sources: list = []
    tokens: list[str] = []

    async for kind, payload in astream_chat(
        question, history, retriever=retriever, llm=llm
    ):
        if kind == "sources":
            sources = payload
        else:
            tokens.append(payload)

    return {"answer": "".join(tokens), "sources": sources}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/rag/test_chain.py -v`
Expected: 10 passed.

- [ ] **Step 5: Run the whole unit suite**

Run: `uv run pytest -q`
Expected: everything passes, still with no infrastructure running.

- [ ] **Step 6: Commit**

```bash
git add app/rag/chain.py tests/rag/test_chain.py
git commit -m "feat: add the grounded RAG chain with streaming and refusal path"
```

---

### Task 7: Wire schemas and the non-streaming endpoint

**Files:**
- Modify: `app/schemas/chat.py`
- Modify: `app/api/dependencies.py`
- Modify: `app/api/chat.py`
- Test: `tests/api/test_chat_endpoint.py`

**Interfaces:**
- Consumes: `achat`, `astream_chat`, `settings.MAX_HISTORY_TURNS`, `settings.GENERATION_TOP_K`, existing `SearchHit`.
- Produces: `ChatMessage`, `ChatRequest`, `ChatResponse`, `get_chat_fn`, `get_stream_fn`, `POST /api/chat`.

- [ ] **Step 1: Write the failing test**

Create `tests/api/test_chat_endpoint.py`:

```python
"""The chat endpoint, with the chain replaced.

Same seam as tests/api/test_chat.py uses for /api/search: the endpoint holds no
logic worth testing except validation and shape, so the chain is overridden and
neither Ollama nor Postgres is contacted.
"""

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.api.dependencies import get_chat_fn
from app.config import settings
from app.main import app

DOC = Document(
    page_content="The fee is BDT 6500 per credit.",
    metadata={
        "ai_content_id": 42,
        "post_id": 7,
        "post_title": "CSE Tuition Fees",
        "chunk_index": 0,
        "score": 0.032258,
        "vector_rank": 1,
        "text_rank": None,
    },
)


@pytest.fixture
def client():
    async def fake_chat(question, history, **kwargs):
        return {"answer": "BDT 6500 per credit. [1]", "sources": [DOC]}

    app.dependency_overrides[get_chat_fn] = lambda: fake_chat
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_an_answer_comes_back_with_its_sources(client):
    response = client.post("/api/chat", json={"question": "What is the CSE fee?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "BDT 6500 per credit. [1]"
    assert body["sources"][0]["post_title"] == "CSE Tuition Fees"
    assert body["sources"][0]["chunk_text"] == "The fee is BDT 6500 per credit."


def test_branch_provenance_survives_to_the_client(client):
    """Same debugging handle /api/search exposes: which branch found this, and
    which one missed it. None must stay None rather than becoming 0."""
    response = client.post("/api/chat", json={"question": "What is the CSE fee?"})

    source = response.json()["sources"][0]
    assert source["vector_rank"] == 1
    assert source["text_rank"] is None


def test_an_empty_question_is_rejected(client):
    """Retrieving on "" burns an embedding call and matches arbitrary chunks."""
    assert client.post("/api/chat", json={"question": ""}).status_code == 422


def test_an_over_long_history_is_rejected(client):
    """History goes into the prompt verbatim. Unbounded history is unbounded
    prompt growth, and on a local model that degrades into a timeout."""
    history = [{"role": "user", "content": "hi"}] * (settings.MAX_HISTORY_TURNS * 2 + 1)

    response = client.post(
        "/api/chat", json={"question": "and for BBA?", "history": history}
    )

    assert response.status_code == 422


def test_a_valid_history_is_accepted(client):
    response = client.post(
        "/api/chat",
        json={
            "question": "and for BBA?",
            "history": [
                {"role": "user", "content": "What is the CSE fee?"},
                {"role": "assistant", "content": "BDT 6500."},
            ],
        },
    )

    assert response.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/api/test_chat_endpoint.py -v`
Expected: FAIL — `ImportError: cannot import name 'get_chat_fn'`

- [ ] **Step 3: Add the schemas**

Append to `app/schemas/chat.py`:

```python
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    # Bounded because history is interpolated into the prompt verbatim.
    history: list[ChatMessage] = Field(
        default_factory=list, max_length=settings.MAX_HISTORY_TURNS * 2
    )
    top_k: int = Field(default=settings.GENERATION_TOP_K, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
    # SearchHit reused rather than a parallel model: the fields are identical,
    # and two shapes for one concept drift the moment either endpoint changes.
    sources: list[SearchHit]
```

Add to the imports at the top of that file:

```python
from typing import Literal

from app.config import settings
```

- [ ] **Step 4: Add the Document → SearchHit conversion**

Append to `app/schemas/chat.py`:

```python
def hit_from_document(doc) -> SearchHit:
    """The wire shape for one retrieved chunk.

    Lives here, not in app/rag/, because app/rag/ must not import the wire
    contract -- keeping that arrow one-way is what lets retrieval be tested
    without FastAPI and the endpoints be tested without retrieval.
    """
    return SearchHit(
        post_id=doc.metadata["post_id"],
        post_title=doc.metadata["post_title"],
        chunk_index=doc.metadata["chunk_index"],
        chunk_text=doc.page_content,
        score=doc.metadata["score"],
        vector_rank=doc.metadata.get("vector_rank"),
        text_rank=doc.metadata.get("text_rank"),
    )
```

- [ ] **Step 5: Add the dependency seams**

Append to `app/api/dependencies.py`:

```python
def get_chat_fn() -> Callable:
    """The non-streaming RAG flow. Overridable, like get_search."""
    from app.rag.chain import achat

    return achat


def get_stream_fn() -> Callable:
    """The streaming RAG flow."""
    from app.rag.chain import astream_chat

    return astream_chat
```

- [ ] **Step 6: Add the endpoint**

Append to `app/api/chat.py`:

```python
@router.post("/chat", response_model=ChatResponse)
async def run_chat(
    payload: ChatRequest,
    chat: Annotated[Callable, Depends(get_chat_fn)],
) -> ChatResponse:
    result = await chat(
        payload.question,
        [(message.role, message.content) for message in payload.history],
    )
    return ChatResponse(
        answer=result["answer"],
        sources=[hit_from_document(doc) for doc in result["sources"]],
    )
```

Extend that file's imports:

```python
from collections.abc import Callable

from app.api.dependencies import SearchFn, get_chat_fn, get_search
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    hit_from_document,
)
```

- [ ] **Step 7: Run test to verify it passes**

Run: `uv run pytest tests/api/test_chat_endpoint.py -v`
Expected: 5 passed.

- [ ] **Step 8: Confirm the existing search endpoint still works**

Run: `uv run pytest tests/api/ -v`
Expected: the `test_chat.py` tests still pass unchanged.

- [ ] **Step 9: Commit**

```bash
git add app/schemas/chat.py app/api/dependencies.py app/api/chat.py tests/api/test_chat_endpoint.py
git commit -m "feat: add POST /api/chat"
```

---

### Task 8: The streaming endpoint

**Files:**
- Modify: `app/api/chat.py`
- Test: `tests/api/test_chat_stream.py`

**Interfaces:**
- Consumes: `get_stream_fn`, `hit_from_document`.
- Produces: `POST /api/chat/stream`, `text/event-stream`.

- [ ] **Step 1: Write the failing test**

Create `tests/api/test_chat_stream.py`:

```python
"""The SSE frame contract the browser depends on.

Order is the contract, not an accident: sources arrive before any token, so the
UI can render citations while the answer is still arriving and a mid-stream
failure still leaves the user holding the evidence.
"""

import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.api.dependencies import get_stream_fn
from app.main import app

DOC = Document(
    page_content="The fee is BDT 6500 per credit.",
    metadata={
        "ai_content_id": 42,
        "post_id": 7,
        "post_title": "CSE Tuition Fees",
        "chunk_index": 0,
        "score": 0.032258,
        "vector_rank": 1,
        "text_rank": None,
    },
)


@pytest.fixture
def client():
    async def fake_stream(question, history, **kwargs):
        yield "sources", [DOC]
        yield "token", "BDT 6500 "
        yield "token", "per credit. [1]"

    app.dependency_overrides[get_stream_fn] = lambda: fake_stream
    yield TestClient(app)
    app.dependency_overrides.clear()


def frames(response):
    return [
        json.loads(line[len("data: ") :])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def test_sources_arrive_before_any_token(client):
    response = client.post("/api/chat/stream", json={"question": "What is the CSE fee?"})

    assert response.status_code == 200
    parsed = frames(response)
    assert parsed[0]["type"] == "sources"
    assert parsed[0]["sources"][0]["post_title"] == "CSE Tuition Fees"


def test_tokens_then_a_single_done(client):
    parsed = frames(client.post("/api/chat/stream", json={"question": "fee?"}))

    assert [frame["type"] for frame in parsed] == ["sources", "token", "token", "done"]
    assert "".join(f["text"] for f in parsed if f["type"] == "token") == (
        "BDT 6500 per credit. [1]"
    )


def test_the_content_type_is_an_event_stream(client):
    response = client.post("/api/chat/stream", json={"question": "fee?"})

    assert response.headers["content-type"].startswith("text/event-stream")


def test_a_failure_mid_stream_becomes_an_error_frame():
    """A raised exception after headers are sent cannot become a 500 -- the
    status line is already gone. The client must be told in-band instead."""

    async def exploding_stream(question, history, **kwargs):
        yield "sources", []
        raise RuntimeError("ollama died")

    app.dependency_overrides[get_stream_fn] = lambda: exploding_stream
    try:
        parsed = frames(TestClient(app).post("/api/chat/stream", json={"question": "x"}))
    finally:
        app.dependency_overrides.clear()

    assert parsed[-1]["type"] == "error"
    assert "done" not in [frame["type"] for frame in parsed]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/api/test_chat_stream.py -v`
Expected: FAIL — 404, the route does not exist.

- [ ] **Step 3: Add the streaming endpoint**

Append to `app/api/chat.py`:

```python
def _sse(payload: dict) -> str:
    """One SSE frame. ensure_ascii=False keeps Bangla readable on the wire."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
async def run_chat_stream(
    payload: ChatRequest,
    stream: Annotated[Callable, Depends(get_stream_fn)],
) -> StreamingResponse:
    history = [(message.role, message.content) for message in payload.history]

    async def frames() -> AsyncIterator[str]:
        try:
            async for kind, data in stream(payload.question, history):
                if kind == "sources":
                    yield _sse(
                        {
                            "type": "sources",
                            "sources": [
                                hit_from_document(doc).model_dump() for doc in data
                            ],
                        }
                    )
                else:
                    yield _sse({"type": "token", "text": data})
        except Exception as exc:
            # The 200 and its headers are already on the wire, so this cannot
            # become a 500. The client is told in-band or not at all.
            yield _sse({"type": "error", "detail": str(exc)})
            return
        yield _sse({"type": "done"})

    return StreamingResponse(
        frames(),
        media_type="text/event-stream",
        # Without this an intervening proxy buffers the whole answer and
        # streaming silently degrades to a long pause then a wall of text.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

Extend that file's imports:

```python
import json
from collections.abc import AsyncIterator, Callable

from fastapi.responses import StreamingResponse

from app.api.dependencies import get_stream_fn
```

Merge `get_stream_fn` into the existing `app.api.dependencies` import line
rather than adding a second import from the same module.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/api/test_chat_stream.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add app/api/chat.py tests/api/test_chat_stream.py
git commit -m "feat: stream chat answers over SSE"
```

---

### Task 9: The chatbot UI

**Files:**
- Create: `app/static/chat.html`
- Modify: `app/main.py`

**Interfaces:**
- Consumes: `POST /api/chat/stream`, `GET /api/health`, `settings.CORPUS_IS_SYNTHETIC`, `settings.MAX_HISTORY_TURNS`.
- Produces: `GET /chat`.

- [ ] **Step 1: Add the route**

In `app/main.py`, add below the existing `dashboard` route:

```python
@app.get("/chat")
def chat_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "chat.html")


@app.get("/api/ui-config")
def ui_config() -> dict:
    """What the page needs to know that only the server knows."""
    return {
        "corpus_is_synthetic": settings.CORPUS_IS_SYNTHETIC,
        "max_history_turns": settings.MAX_HISTORY_TURNS,
    }
```

Add to that file's imports:

```python
from app.config import settings
```

- [ ] **Step 2: Create the page**

Create `app/static/chat.html`:

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DIU Admission Assistant</title>
<style>
  :root {
    --bg: #f6f7f9; --panel: #fff; --ink: #16181d; --muted: #6b7280;
    --line: #e3e6ea; --accent: #1f6feb; --warn-bg: #fff8e6; --warn-ink: #9a6700;
    --bubble-user: #1f6feb; --bubble-user-ink: #fff; --radius: 10px;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0f1115; --panel: #171a21; --ink: #e8eaed; --muted: #9aa3af;
      --line: #262b35; --accent: #4c8dff; --warn-bg: #2b2413; --warn-ink: #d29922;
      --bubble-user: #2d5fd0; --bubble-user-ink: #fff;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink); height: 100vh;
    display: flex; flex-direction: column;
    font: 14px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans Bengali", sans-serif;
  }
  header {
    background: var(--panel); border-bottom: 1px solid var(--line);
    padding: 14px 22px; display: flex; align-items: center; gap: 12px;
  }
  h1 { font-size: 16px; margin: 0; font-weight: 650; }
  header a { margin-left: auto; color: var(--muted); font-size: 12px; }
  .banner {
    background: var(--warn-bg); color: var(--warn-ink); padding: 9px 22px;
    font-size: 12.5px; border-bottom: 1px solid var(--line);
  }
  #log { flex: 1; overflow-y: auto; padding: 22px; max-width: 780px; margin: 0 auto; width: 100%; }
  .msg { margin-bottom: 18px; display: flex; }
  .msg.user { justify-content: flex-end; }
  .bubble {
    padding: 10px 14px; border-radius: var(--radius); max-width: 78%;
    white-space: pre-wrap; overflow-wrap: anywhere;
  }
  .user .bubble { background: var(--bubble-user); color: var(--bubble-user-ink); }
  .bot .bubble { background: var(--panel); border: 1px solid var(--line); }
  .cites { margin-top: 9px; display: flex; flex-wrap: wrap; gap: 6px; }
  .cite {
    font-size: 11.5px; border: 1px solid var(--line); background: var(--bg);
    color: var(--muted); border-radius: 999px; padding: 3px 10px; cursor: pointer;
  }
  .cite:hover { border-color: var(--accent); color: var(--accent); }
  .chunk {
    margin-top: 8px; font-size: 12.5px; color: var(--muted);
    border-left: 2px solid var(--line); padding: 4px 0 4px 10px; white-space: pre-wrap;
  }
  .empty { color: var(--muted); text-align: center; margin-top: 12vh; }
  .example {
    display: block; margin: 8px auto; background: var(--panel); color: var(--ink);
    border: 1px solid var(--line); border-radius: 999px; padding: 7px 15px;
    cursor: pointer; font: inherit;
  }
  .example:hover { border-color: var(--accent); }
  footer { border-top: 1px solid var(--line); background: var(--panel); padding: 14px 22px; }
  form { display: flex; gap: 10px; max-width: 780px; margin: 0 auto; }
  input {
    flex: 1; padding: 11px 14px; border: 1px solid var(--line); border-radius: 8px;
    background: var(--bg); color: var(--ink); font: inherit;
  }
  button[type=submit] {
    border: 1px solid var(--accent); background: var(--accent); color: #fff;
    border-radius: 8px; padding: 9px 18px; font: inherit; cursor: pointer;
  }
  button[disabled] { opacity: .5; cursor: not-allowed; }
</style>
</head>
<body>
<header>
  <h1>DIU Admission Assistant</h1>
  <a href="/">Ingestion dashboard →</a>
</header>
<div class="banner" id="banner" hidden></div>

<div id="log">
  <div class="empty" id="empty">
    <p>Ask about admission at Daffodil International University.</p>
    <button class="example">What is the CSE tuition fee per credit?</button>
    <button class="example">সিএসই ভর্তি ফি কত টাকা?</button>
  </div>
</div>

<footer>
  <form id="form">
    <input id="q" autocomplete="off" placeholder="Ask a question…  /  একটি প্রশ্ন করুন…" required>
    <button type="submit" id="send">Send</button>
  </form>
</footer>

<script>
const log = document.getElementById('log');
const form = document.getElementById('form');
const input = document.getElementById('q');
const send = document.getElementById('send');
const banner = document.getElementById('banner');

// The transcript the server never sees between requests: this page is the
// only place conversation state lives, which is why the API stays stateless.
let history = [];
let maxTurns = 6;

fetch('/api/ui-config').then(r => r.json()).then(cfg => {
  maxTurns = cfg.max_history_turns;
  if (cfg.corpus_is_synthetic) {
    banner.hidden = false;
    banner.textContent =
      '⚠ Sample data. Every fee, deadline and GPA figure here is invented for ' +
      'testing and is NOT real DIU information.';
  }
});

function bubble(role, text) {
  document.getElementById('empty')?.remove();
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + role;
  const b = document.createElement('div');
  b.className = 'bubble';
  b.textContent = text;
  wrap.appendChild(b);
  log.appendChild(wrap);
  log.scrollTop = log.scrollHeight;
  return b;
}

function renderCitations(bubbleEl, sources) {
  if (!sources.length) return;
  const row = document.createElement('div');
  row.className = 'cites';
  sources.forEach((s, i) => {
    const chip = document.createElement('button');
    chip.className = 'cite';
    chip.textContent = `[${i + 1}] ${s.post_title}`;
    chip.onclick = () => {
      const open = bubbleEl.parentElement.querySelector(`[data-chunk="${i}"]`);
      if (open) { open.remove(); return; }
      const pre = document.createElement('div');
      pre.className = 'chunk';
      pre.dataset.chunk = i;
      pre.textContent = s.chunk_text;
      bubbleEl.parentElement.appendChild(pre);
    };
    row.appendChild(chip);
  });
  bubbleEl.parentElement.appendChild(row);
}

async function ask(question) {
  bubble('user', question);
  const out = bubble('bot', '');
  send.disabled = input.disabled = true;

  let answer = '';
  try {
    const res = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, history: history.slice(-maxTurns * 2) }),
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // Frames are \n\n-delimited; a chunk may split one, so keep the tail.
      const parts = buffer.split('\n\n');
      buffer = parts.pop();
      for (const part of parts) {
        if (!part.startsWith('data: ')) continue;
        const frame = JSON.parse(part.slice(6));
        if (frame.type === 'sources') renderCitations(out, frame.sources);
        else if (frame.type === 'token') { answer += frame.text; out.textContent = answer; }
        else if (frame.type === 'error') { out.textContent = answer || ('Error: ' + frame.detail); }
        log.scrollTop = log.scrollHeight;
      }
    }
  } catch (err) {
    out.textContent = 'Could not reach the assistant: ' + err.message;
  } finally {
    send.disabled = input.disabled = false;
    input.focus();
  }

  if (answer) {
    history.push({ role: 'user', content: question },
                 { role: 'assistant', content: answer });
    history = history.slice(-maxTurns * 2);
  }
}

form.onsubmit = (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  input.value = '';
  ask(question);
};

document.addEventListener('click', (e) => {
  if (e.target.classList.contains('example')) ask(e.target.textContent);
});
</script>
</body>
</html>
```

- [ ] **Step 3: Verify the routes serve**

Run: `uv run pytest -q` (nothing should regress), then start the app and check by hand:

```bash
uv run uvicorn app.main:app --port 8000
```

Expected: `GET /chat` returns the page; `GET /api/ui-config` returns JSON with `corpus_is_synthetic: true`.

- [ ] **Step 4: Commit**

```bash
git add app/static/chat.html app/main.py
git commit -m "feat: add the streaming chatbot UI at /chat"
```

---

### Task 10: Integration tests against a real model

**Files:**
- Modify: `tests/conftest.py`
- Create: `tests/integration/test_generation.py`

**Interfaces:**
- Consumes: everything above, plus a running Ollama, Postgres and embedding service.
- Produces: the `ollama` fixture.

**Note:** agent Bash tooling blocks `127.0.0.1:11434`, so these skip inside the sandbox and must be run with the sandbox disabled.

- [ ] **Step 1: Add the fixture**

Append to `tests/conftest.py`:

```python
@pytest.fixture(scope="session")
def ollama():
    """Skip unless Ollama is up, with the same fast socket probe as the rest."""
    url = urlparse(settings.OLLAMA_BASE_URL)
    host, port = url.hostname or "127.0.0.1", url.port or 11434
    if not _reachable(host, port):
        pytest.skip(
            f"Ollama not reachable on {host}:{port} "
            f"(ollama serve; ollama pull {settings.OLLAMA_MODEL})"
        )
    return settings.OLLAMA_MODEL
```

- [ ] **Step 2: Write the integration tests**

Create `tests/integration/test_generation.py`:

```python
"""Generation against the real model, with retrieval faked.

Retrieval quality is covered by tests/test_retrieval_smoke.py. What is checked
here is what only a real model can answer: does it stay inside the context, does
it answer in the right script, and does it decline instead of inventing a figure.
Every one of these assertions failed against at least one model during design.
"""

import pytest

from app.rag.chain import achat
from app.rag.langchain_retriever import HybridRetriever
from app.rag.language import REFUSALS
from app.rag.retriever import Hit

pytestmark = pytest.mark.integration

FEE = Hit(
    ai_content_id=1,
    post_id=1,
    post_title="Tuition Fees (CSE)",
    chunk_index=0,
    chunk_text="The CSE tuition fee is BDT 6,500 per credit.",
    score=0.03,
    vector_rank=1,
    text_rank=1,
)
# Real chunks that do not answer a hostel question. This, not an empty list, is
# what retrieval actually returns for an unanswerable question: hybrid search
# returns the nearest k chunks however distant they are.
IRRELEVANT = Hit(
    ai_content_id=2,
    post_id=2,
    post_title="Admission Deadlines",
    chunk_index=0,
    chunk_text="Applications for the Spring semester close on 30 June.",
    score=0.02,
    vector_rank=1,
    text_rank=1,
)


def _retriever(*hits):
    return HybridRetriever(search_fn=lambda query, top_k: list(hits), top_k=5)


def is_bengali(text: str) -> bool:
    return any("ঀ" <= char <= "৿" for char in text)


async def test_an_english_question_is_answered_in_english(ollama):
    result = await achat(
        "What is the CSE tuition fee per credit?", [], retriever=_retriever(FEE)
    )

    assert "6,500" in result["answer"] or "6500" in result["answer"]
    # gemma3:4b answered English questions in Bangla until the language rule
    # was injected explicitly rather than inferred.
    assert not is_bengali(result["answer"])


async def test_a_bangla_script_question_is_answered_in_bangla(ollama):
    result = await achat("সিএসই ভর্তি ফি কত টাকা?", [], retriever=_retriever(FEE))

    assert is_bengali(result["answer"])
    # Rule 2 also demands Western digits, so the figure stays greppable.
    assert "6,500" in result["answer"] or "6500" in result["answer"]


@pytest.mark.parametrize(
    "question",
    [
        "CSE er tuition fee koto",
        "amar CSE te vorti hote koto taka lagbe",
        "vorti fee koto taka",
    ],
)
async def test_a_banglish_question_is_answered_in_bengali_script(ollama, question):
    """The hard requirement, end to end.

    Told only "answer in Bangla", the model answered 2 of 3 of these in English:
    it anchors on the question's Latin script and overrides the instruction.
    Naming the script and giving a one-shot example is what fixed it, so this
    test is what catches anyone trimming that rule.
    """
    result = await achat(question, [], retriever=_retriever(FEE))

    assert is_bengali(result["answer"])
    assert "6,500" in result["answer"] or "6500" in result["answer"]


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the hostel fee?", REFUSALS["English"]),
        ("hostel fee koto", REFUSALS["Bangla"]),
        ("হোস্টেল ফি কত?", REFUSALS["Bangla"]),
    ],
)
async def test_an_unanswerable_question_is_refused_not_invented(
    ollama, question, expected
):
    """The failure this whole design exists to prevent: a confident, fluent,
    invented fee. Note the context is real but irrelevant -- given an *empty*
    context the same model invented "$360 per year" and "BDT 18,000"."""
    result = await achat(question, [], retriever=_retriever(FEE, IRRELEVANT))

    assert result["answer"] == expected
```

- [ ] **Step 3: Run the integration tests**

Run: `uv run pytest -m integration tests/integration/test_generation.py -v`
Expected: 8 passed with Ollama running; 8 skipped with it stopped.

Note: agent Bash tooling blocks `127.0.0.1:11434`, so this needs the sandbox disabled.

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py tests/integration/test_generation.py
git commit -m "test: verify grounded bilingual generation against a live model"
```

---

### Task 11: Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Move generation out of the scope limits**

In `README.md`, under **Deliberate scope limits**, delete the bullet beginning `**No answer generation yet.**` — it is no longer true.

- [ ] **Step 2: Document the new stack rows**

Add to the **Stack** table:

```markdown
| Generation | **gemma3:4b** via Ollama, local | Keeps the no-API-keys property. Chosen by probe: the alternative tested could not produce Bangla at all |
| Orchestration | `langchain-core` 1.x | Prompt/LLM/retriever composition and streaming only — retrieval stays hand-rolled |
```

- [ ] **Step 3: Document setup and the new endpoints**

Add to **Setup**, after the embedding service step:

````markdown
```bash
# 5. Generation model (~3.3 GB, then works offline)
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
````

- [ ] **Step 4: Record what generation does and does not guarantee**

Add a subsection under **Deliberate scope limits**:

```markdown
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
```

- [ ] **Step 5: Run everything**

Run: `uv run pytest -q` then, outside the sandbox, `uv run pytest -m integration -q`
Expected: unit tests all pass; integration tests pass with infrastructure up.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: document generation, the chat UI, and its guarantees"
```
