# প্রজেক্ট ফাইল স্ট্রাকচার (বাংলা)

এই প্রজেক্টের পুরো ফোল্ডার-ফাইল স্ট্রাকচার, প্রতিটা অংশ কী কাজ করে তার সংক্ষিপ্ত
ব্যাখ্যাসহ। টুলিং/ক্যাশ ফোল্ডার (`.venv`, `.git`, `__pycache__`, `.ruff_cache`,
`.agents`, `.claude`, `.superpowers`) বাদ দেওয়া হয়েছে — এগুলো কোডের অংশ না।

```
DIU_Admission_RAG/
│
├── app/                              ← মূল অ্যাপ্লিকেশন কোড
│   ├── main.py                       FastAPI অ্যাপ: router + static mount, আর কিছু না
│   │
│   ├── api/                          ← HTTP লেয়ার — validate, dependency কল, status code
│   │   ├── __init__.py
│   │   ├── admin.py                  GET /api/health (DB + embedding service আলাদা করে চেক করে)
│   │   ├── chat.py                   POST /api/search, /api/chat, /api/chat/stream
│   │   ├── documents.py              পোস্ট লাইফসাইকেল: create/publish/approve/verify/ingest
│   │   └── dependencies.py           FastAPI Depends() — repo, search fn, chat fn ইত্যাদি
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py               সব কনফিগের একমাত্র উৎস (DB URL, মডেল, চাংক সাইজ ইত্যাদি)
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py             psycopg3 connection pool (get_conn, get_pool)
│   │   └── repositories.py           সব SQL এখানে — PostRepository ক্লাস
│   │
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── service.py                BGE-M3 মডেল সার্ভ করা আলাদা FastAPI মাইক্রোসার্ভিস (পোর্ট 8001)
│   │   └── client.py                 মূল অ্যাপ থেকে ওই সার্ভিসকে HTTP দিয়ে কল করার ক্লায়েন্ট
│   │
│   ├── ingestion/                    ← RAG পাইপলাইন: chunk → embed → store
│   │   ├── __init__.py
│   │   ├── chunker.py                পোস্টের বডি ভেঙে chunk বানানো (টোকেন-ভিত্তিক)
│   │   ├── pipeline.py               ingest_post() — পুরো পাইপলাইনের মূল লজিক
│   │   └── run.py                    CLI: `python -m app.ingestion.run --all-eligible`
│   │
│   ├── llm/
│   │   └── __init__.py               get_chat_model() — Ollama/ChatOllama তৈরির একমাত্র জায়গা
│   │
│   ├── rag/                          ← রিট্রিভাল + জেনারেশন লজিক
│   │   ├── __init__.py
│   │   ├── cosine_search.py          ভেক্টর (semantic) সার্চ ব্রাঞ্চ
│   │   ├── bm25_search.py            লেক্সিক্যাল (full-text) সার্চ ব্রাঞ্চ
│   │   ├── hybrid_search.py          দুটো ব্রাঞ্চ ফিউজ করা (Reciprocal Rank Fusion)
│   │   ├── retriever.py              cosine + bm25 + hybrid একসাথে জোড়া দেওয়া
│   │   ├── langchain_retriever.py    retriever-কে LangChain-এর ইন্টারফেসে মানিয়ে নেওয়া
│   │   ├── language.py               উত্তর কোন ভাষায় (বাংলা/ইংরেজি/বাংলিশ) হবে তা ঠিক করা + রিফিউজাল
│   │   ├── prompts.py                LLM-কে দেওয়া প্রম্পট টেমপ্লেট
│   │   ├── chain.py                  পুরো RAG চেইন (achat / astream_chat)
│   │   └── rows.py                   DB row থেকে পাইথন অবজেক্টে রূপান্তরের হেল্পার
│   │
│   ├── schemas/                      ← Pydantic মডেল (request/response এর ওয়্যার কন্ট্র্যাক্ট)
│   │   ├── __init__.py
│   │   ├── chat.py                   চ্যাট/সার্চ রিকোয়েস্ট-রেসপন্স স্কিমা
│   │   └── document.py               পোস্ট/চাংক/ফেইলিওর স্কিমা
│   │
│   └── static/                       ← ফ্রন্টএন্ড (plain HTML+JS, কোনো ফ্রেমওয়ার্ক নেই)
│       ├── index.html                ড্যাশবোর্ড: পোস্ট তৈরি/approve/verify/publish, চাংক দেখা
│       └── chat.html                 চ্যাটবট UI
│
├── migrations/
│   └── 001_schema.sql                সম্পূর্ণ ডাটাবেজ স্কিমা — একমাত্র সত্যের উৎস (source of truth)
│
├── scripts/
│   ├── setup_db.sh                   Postgres+pgvector ইনস্টল/চালু/স্কিমা প্রয়োগ — নিরাপদে re-run করা যায়
│   └── seed.py                       ১৫টা কাল্পনিক sample পোস্ট + একজন seed user ঢোকায় (শুধু ডেমো/টেস্টের জন্য)
│
├── tests/                            ← pytest টেস্ট
│   ├── conftest.py                   fixtures (in-memory repo override ইত্যাদি)
│   ├── api/
│   │   ├── test_chat.py
│   │   ├── test_chat_endpoint.py
│   │   ├── test_chat_stream.py
│   │   ├── test_documents.py         documents.py এন্ডপয়েন্টের টেস্ট
│   │   └── test_pages.py             static পেজ সার্ভ হচ্ছে কিনা
│   ├── ingestion/
│   │   └── test_chunker.py
│   ├── integration/
│   │   └── test_generation.py        আসল Ollama লাগে — না থাকলে skip হয়
│   ├── rag/
│   │   ├── test_chain.py
│   │   ├── test_hybrid_search.py
│   │   ├── test_langchain_retriever.py
│   │   ├── test_language.py
│   │   └── test_prompts.py
│   ├── test_llm_factory.py
│   └── test_retrieval_smoke.py       আসল DB+embedding লাগে — রিট্রিভাল কোয়ালিটির প্রমাণ (10/10 প্রশ্ন)
│
├── docs/                             ← ডকুমেন্টেশন
│   ├── agents/                       Agent skill/workflow ডক (issue tracker, triage labels, domain)
│   │   ├── domain.md
│   │   ├── issue-tracker.md
│   │   └── triage-labels.md
│   ├── superpowers/                  ডিজাইন স্পেক ও প্ল্যান (langchain-chat ফিচারের)
│   │   ├── plans/2026-09-05-langchain-chat.md
│   │   └── specs/2026-09-05-langchain-rag-chat-design.md
│   ├── schema-change-requests.md     DB owner-কে পাঠানো schema পরিবর্তনের অনুরোধের তালিকা
│   ├── user-guide-bn.md              ★ দৈনন্দিন ব্যবহার গাইড (সার্ভার চালু, ডেটা তৈরি, troubleshooting)
│   └── database-and-precautions-bn.md ★ DB কানেকশন, সতর্কতা, do's/don'ts, dependency map
│
├── CLAUDE.md                          এই রিপোতে কাজ করার এজেন্ট-নির্দিষ্ট নির্দেশনা
├── README.md                          মূল টেকনিক্যাল ডকুমেন্টেশন (ইংরেজি, স্ট্যাক/আর্কিটেকচার/স্কোপ)
├── .env.example                       সব কনফিগের ডিফল্ট মান (কপি করে `.env` বানাতে হয়)
├── pyproject.toml                     পাইথন ডিপেন্ডেন্সি ও প্রজেক্ট মেটাডেটা (uv দিয়ে ম্যানেজড)
└── uv.lock                            লক করা ডিপেন্ডেন্সি ভার্সন
```

---

## স্তরভিত্তিক সংক্ষিপ্ত মানচিত্র

```
রিকোয়েস্ট আসার পথ:

ব্রাউজার
  │
  ▼
app/static/*.html  (index.html / chat.html)
  │  fetch() কল করে
  ▼
app/api/*.py        (admin / chat / documents) ── HTTP validate + status code
  │  Depends() দিয়ে
  ▼
app/api/dependencies.py
  │
  ├──► app/database/repositories.py ──► app/database/connection.py ──► PostgreSQL (5433)
  │
  ├──► app/rag/retriever.py ──► cosine_search.py + bm25_search.py ──► hybrid_search.py
  │         │
  │         └──► app/embeddings/client.py ──► app/embeddings/service.py (৮০০১ পোর্ট, আলাদা প্রসেস)
  │
  └──► app/rag/chain.py ──► app/llm/ (Ollama) ──► উত্তর তৈরি

ইনজেস্ট পাইপলাইন (verify করলে ব্যাকগ্রাউন্ডে চলে):
app/ingestion/pipeline.py
  → app/ingestion/chunker.py    (বডি ভাঙা)
  → app/embeddings/client.py    (এমবেড করা)
  → app/database/repositories.py (ai_content + ai_content_vectors-এ সেভ)
```

## মূল নিয়ম (README থেকে, মনে রাখার মতো)

- **সব SQL `app/database/repositories.py`-তে থাকে, কখনো এন্ডপয়েন্টে সরাসরি লেখা হয় না।**
- **র‍্যাংকিং লজিক (`app/rag/hybrid_search.py`) কোনো I/O করে না** — তাই ডাটাবেজ ছাড়াই এটা টেস্ট করা যায়।
- **`migrations/001_schema.sql`-ই ডাটাবেজ স্কিমার একমাত্র সত্যি উৎস** — সরাসরি psql দিয়ে টেবিল বদলালে এই ফাইলের সাথে বাস্তবতা মিলবে না।
