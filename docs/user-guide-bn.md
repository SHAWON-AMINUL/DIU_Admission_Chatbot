# DIU Admission RAG — ব্যবহারকারীর গাইড (বাংলা)

এই ডকুমেন্টটা দৈনন্দিন কাজের জন্য: সার্ভার চালু করা, ডাটাবেজ সেটআপ/খালি করা,
ইউজার-পোস্ট তৈরি করে RAG পাইপলাইন চালানো, এবং কমন সমস্যাগুলোর সমাধান।
টেকনিক্যাল স্ট্যাক ও আর্কিটেকচারের বিস্তারিত জানতে ইংরেজি [README.md](../README.md) দেখুন।

---

## ১. প্রজেক্টের অংশগুলো

এই সিস্টেমে একসাথে **তিনটা জিনিস চালু** রাখতে হয়:

| # | জিনিস | পোর্ট | কমান্ড |
|---|---|---|---|
| 1 | **PostgreSQL** (pgvector সহ) | 5433 | `brew services start postgresql@18` |
| 2 | **Embedding service** (BGE-M3 মডেল) | 8001 | `uv run uvicorn app.embeddings.service:app --port 8001` |
| 3 | **মূল API + ড্যাশবোর্ড** | 8000 | `uv run uvicorn app.main:app --reload --port 8000` |

> ⚠️ পোর্ট **5433**, 5432 না — কারণ এই মেশিনে আগে থেকেই একটা EnterpriseDB Postgres 5432-এ চলে, সেটাকে না ছুঁয়ে এই প্রজেক্ট আলাদা পোর্টে নিজের Postgres চালায়।

চ্যাটে জেনারেশনের জন্য Ollama (gemma3:4b) আলাদাভাবে দরকার হয়, কিন্তু শুধু ডেটা ঢোকানো/embedding পরীক্ষার জন্য উপরের তিনটাই যথেষ্ট।

---

## ২. প্রথমবার সেটআপ (একবারই করতে হয়)

```bash
# ভার্চুয়াল এনভায়রনমেন্ট চালু করুন
source .venv/bin/activate

# ডাটাবেজ তৈরি (role + db + extension + schema) — নিরাপদ, বারবার চালানো যায়
brew install postgresql@18 pgvector
bash scripts/setup_db.sh

# পাইথন ডিপেন্ডেন্সি
uv sync

# (ঐচ্ছিক) চ্যাট জেনারেশনের জন্য
brew install ollama
ollama serve &
ollama pull gemma3:4b
```

`scripts/setup_db.sh` স্ক্রিপ্টটাই টেবিল তৈরি করে ([migrations/001_schema.sql](../migrations/001_schema.sql) থেকে) — এটা re-run করলেও কিছু ডিলিট হয় না, শুধু যা নেই তা বানায়।

---

## ৩. প্রতিদিন সার্ভার চালু করার ধাপ (ক্রম গুরুত্বপূর্ণ)

**ধাপ ১ — Postgres চালু করুন এবং নিশ্চিত হোন এটা রেডি:**

```bash
brew services start postgresql@18
# কয়েক সেকেন্ড অপেক্ষা করুন, তারপর চেক করুন:
/opt/homebrew/opt/postgresql@18/bin/pg_isready -h localhost -p 5433
```

`accepting connections` না লেখা পর্যন্ত পরের ধাপে যাবেন না — Postgres রেডি হওয়ার আগে অ্যাপ চালু করলে `Connection refused` এরর আসবে।

**ধাপ ২ — Embedding service চালু করুন (আলাদা টার্মিনালে):**

```bash
uv run uvicorn app.embeddings.service:app --port 8001
```

লগে `model ready (dim=1024)` না দেখা পর্যন্ত অপেক্ষা করুন (প্রথমবার মডেল লোড হতে একটু সময় নেয়)। **এই টার্মিনাল বন্ধ করবেন না** — যতক্ষণ প্রশ্ন/সার্চ/verify করবেন ততক্ষণ এটা চালু থাকতে হবে।

**ধাপ ৩ — মূল API চালু করুন (আরেকটা আলাদা টার্মিনালে):**

```bash
uv run uvicorn app.main:app --reload --port 8000
```

এখন খুলুন:
- ড্যাশবোর্ড (পোস্ট তৈরি/approve/verify): **http://localhost:8000**
- চ্যাটবট UI: **http://localhost:8000/chat**

---

## ৪. ডাটাবেজ টেবিল খালি করা (রিসেট)

যদি সব ডেটা মুছে নতুন করে শুরু করতে চান:

```bash
PGPASSWORD=diu /opt/homebrew/opt/postgresql@18/bin/psql \
  -h localhost -p 5433 -U diu -d diu_admission \
  -c "TRUNCATE TABLE users, categories, posts, post_files, fail_history, category_post, ai_content, ai_content_vectors RESTART IDENTITY CASCADE;"
```

এটা ৮টা টেবিলই খালি করে এবং id সিরিয়াল ১ থেকে আবার শুরু করে দেয়। **এটা আনডু করা যায় না** — শুধু dev/testing ডেটার জন্য ব্যবহার করুন।

সম্পূর্ণ ডাটাবেজ থেকেই মুছে একদম নতুন বানাতে চাইলে:

```bash
dropdb -h localhost -p 5433 diu_admission
bash scripts/setup_db.sh
```

---

## ৫. User তৈরি করা

⚠️ **এই প্রজেক্টে এখনো signup/login বা "user তৈরি করুন" ফর্ম নেই** — কোনো ফ্রন্টএন্ড বা API endpoint (`POST /api/users` জাতীয় কিছু) নেই। এটা এখনো auth স্লাইসে যোগ হয়নি।

পোস্ট তৈরি করতে হলে `users` টেবিলে অন্তত **একজন ইউজার থাকতেই হবে**, কারণ প্রতিটা পোস্টের একজন `created_by` লাগে। ইউজার না থাকলে `POST /api/posts` কল করলে এই এরর আসবে:

```
"no users exist; run the seed script first"
```

সমাধান — SQL দিয়ে সরাসরি একটা প্লেসহোল্ডার ইউজার বসিয়ে দিন:

```bash
PGPASSWORD=diu /opt/homebrew/opt/postgresql@18/bin/psql \
  -h localhost -p 5433 -U diu -d diu_admission \
  -c "INSERT INTO users (name, email, password, status) VALUES ('Admission Officer (seed)', 'seed@example.invalid', 'NOT-A-REAL-HASH-seed-only', 'active') RETURNING id;"
```

এটা শুধু ডেটাবেজের ভেতরের সম্পর্ক (FK) পূরণের জন্য — আসল লগইন-পাসওয়ার্ড কিছু না। যতক্ষণ `users` টেবিলে একজনও থাকবে, নতুন পোস্ট তৈরি অ্যাপ স্বয়ংক্রিয়ভাবে সেই ইউজারকেই `created_by` ধরবে (`app/database/repositories.py` এর `_any_user_id()`)।

বিকল্প: `uv run python scripts/seed.py` চালালে এটা নিজে থেকেই একটা seed user + ১৫টা sample পোস্ট বানিয়ে দেয় — কিন্তু `posts` টেবিলে আগে থেকে কোনো পোস্ট থাকলে এটা কিছুই করবে না (safety check)।

---

## ৬. একটা পোস্ট তৈরি করে RAG পাইপলাইন সম্পূর্ণ করা

একটা পোস্টকে embedding পর্যন্ত নিয়ে যেতে **৪টা ধাপ** পার হতে হয়:

```
Draft → publish → approve (director) → verify → [ব্যাকগ্রাউন্ডে ইনজেস্ট: chunk + embed]
```

### ড্যাশবোর্ড (http://localhost:8000) থেকে:

1. Title + body লিখে **"Create"** চাপুন → পোস্ট `Draft` অবস্থায় তৈরি হবে।
2. **Publish** বাটন চাপুন।
3. **Approve** বাটন চাপুন (director-level অনুমোদন)।
4. **Verify → Ingest** বাটন চাপুন — এটাই embedding pipeline ট্রিগার করে ব্যাকগ্রাউন্ডে।
5. কয়েক সেকেন্ড পর টেবিলে "chunks" আর "vectors" কলামে সংখ্যা দেখা গেলে বুঝবেন embedding সফল হয়েছে।

### বা টার্মিনাল থেকে (curl দিয়ে):

```bash
# ১. ড্রাফট তৈরি
curl -s -X POST http://127.0.0.1:8000/api/posts \
  -H "Content-Type: application/json" \
  -d '{"title":"Tuition Fees","body":"CSE-এর টিউশন ফি প্রতি ক্রেডিট...","category_ids":[]}'
# → {"id": 1, "status": "Draft"}

# ২. পাবলিশ
curl -s -X POST http://127.0.0.1:8000/api/posts/1/publish

# ৩. অ্যাপ্রুভ
curl -s -X POST http://127.0.0.1:8000/api/posts/1/approve

# ৪. ভেরিফাই (এটাই ইনজেস্ট/এমবেডিং শুরু করে)
curl -s -X POST http://127.0.0.1:8000/api/posts/1/verify
```

### সব eligible পোস্ট একসাথে ইনজেস্ট করতে (CLI):

```bash
uv run python -m app.ingestion.run --all-eligible
# একটা নির্দিষ্ট পোস্ট:
uv run python -m app.ingestion.run --post-id 1
```

---

## ৭. Embedding হয়েছে কিনা কীভাবে যাচাই করবেন

**ড্যাশবোর্ডে:** পোস্ট লিস্টে `chunks` আর `vectors` কলামে সংখ্যা ০ না হয়ে বেশি হলে embedding সম্পন্ন হয়েছে। পোস্টের status কলামে `AI training complete` লেখা থাকবে।

**SQL দিয়ে সরাসরি:**

```bash
PGPASSWORD=diu /opt/homebrew/opt/postgresql@18/bin/psql \
  -h localhost -p 5433 -U diu -d diu_admission \
  -c "SELECT id, title, status, approval_status, verify_status, fail_count FROM posts ORDER BY id;"

# চাংক আর ভেক্টর কতগুলো তৈরি হয়েছে:
PGPASSWORD=diu /opt/homebrew/opt/postgresql@18/bin/psql \
  -h localhost -p 5433 -U diu -d diu_admission \
  -c "SELECT count(*) FROM ai_content; SELECT count(*) FROM ai_content_vectors;"
```

`status = 'AI training complete'` এবং `ai_content_vectors` এ row থাকলে সেই পোস্ট RAG চ্যাটে ব্যবহারযোগ্য।

`status = 'AI training failed'` হলে কারণ দেখুন:

```bash
curl -s http://127.0.0.1:8000/api/posts/1/failures
```

---

## ৮. চ্যাট/সার্চ টেস্ট করা

```bash
# শুধু রিট্রিভাল (কোনো উত্তর জেনারেট হয় না, শুধু matched chunks দেখায়)
curl -s -X POST http://127.0.0.1:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query":"CSE-এর টিউশন ফি কত?","top_k":5}'

# পূর্ণ RAG উত্তর (Ollama লাগবে)
curl -s -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"CSE-এর টিউশন ফি কত?"}'
```

অথবা ব্রাউজারে **http://localhost:8000/chat** খুলে সরাসরি প্রশ্ন করুন।

---

## ৯. Troubleshooting

### ৯.১ `Connection refused` — port 5433 (Postgres)

**উপসর্গ:** অ্যাপ চালু করার সাথে সাথে লগে বারবার `connection to server at "::1", port 5433 failed: Connection refused`।

**কারণ:** Postgres এখনো চালু হয়নি বা অ্যাপ Postgres চালু হওয়ার *আগেই* চালু হয়ে গেছে।

**সমাধান:**
```bash
brew services start postgresql@18
/opt/homebrew/opt/postgresql@18/bin/pg_isready -h localhost -p 5433
```
`accepting connections` দেখা গেলে uvicorn (app.main) restart করুন (`--reload` চালু থাকলে এমনিই সেরে যাবে, নাহলে Ctrl+C দিয়ে আবার রান করুন)।

---

### ৯.২ `/api/search` বা `/api/chat` এ 500 error, লগে `httpx.ConnectError: Connection refused`

**কারণ:** Embedding service (পোর্ট 8001) বন্ধ আছে। রিট্রিভালের জন্য প্রতিটা কোয়েরি এই সার্ভিসে embedding বানাতে যায়।

**সমাধান:** আলাদা টার্মিনালে এটা চালু রাখুন —
```bash
uv run uvicorn app.embeddings.service:app --port 8001
```
"model ready" না আসা পর্যন্ত অপেক্ষা করুন।

---

### ৯.৩ `POST /api/posts` → `"no users exist; run the seed script first"`

**কারণ:** `users` টেবিল খালি।

**সমাধান:** [ধারা ৫](#৫-user-তৈরি-করা) অনুযায়ী একটা user insert করুন, অথবা `uv run python scripts/seed.py` চালান (পোস্ট টেবিল খালি থাকলে)।

---

### ৯.৪ Verify বাটনে ক্লিক করার পর কিছুই হচ্ছে না / অনেকক্ষণ ধরে "loading" / পরে 500 error

**কারণ:** একই পোস্টে **একসাথে একাধিক verify request** গেলে (একাধিক ট্যাব খোলা রাখা, বা টানা কয়েকবার বাটনে ক্লিক করা) — প্রতিটা request পোস্টের row-তে লক নেয়, আর প্রথমটা কখনো commit হতে পারে না (thread-pool/lock contention)। ফলাফল: সবগুলো request চিরকাল আটকে থাকে, শেষে timeout/500 error।

**এড়ানোর উপায়:**
- একটাই ট্যাবে ড্যাশবোর্ড খোলা রাখুন।
- Verify বাটনে ক্লিক করার পর "Verified — ingest running in background" টোস্ট না আসা পর্যন্ত আবার ক্লিক করবেন না।

**যদি আগেই আটকে গিয়ে থাকে — আটকে থাকা connection ভেঙে দিন:**

```bash
PGPASSWORD=diu /opt/homebrew/opt/postgresql@18/bin/psql \
  -h localhost -p 5433 -U diu -d diu_admission \
  -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='diu_admission' AND pid <> pg_backend_pid() AND (state = 'idle in transaction' OR wait_event_type='Lock');"
```

এটা চালানোর পর আটকে থাকা verify request rollback হয়ে যাবে (পোস্ট আগের অবস্থায় ফিরে যাবে) — একটা ট্যাব থেকে আবার একবার verify করুন।

**কোন কানেকশনগুলো আটকে আছে দেখতে:**

```bash
PGPASSWORD=diu /opt/homebrew/opt/postgresql@18/bin/psql \
  -h localhost -p 5433 -U diu -d diu_admission \
  -c "SELECT pid, state, wait_event_type, wait_event, query, now()-query_start AS duration FROM pg_stat_activity WHERE datname='diu_admission';"
```

---

### ৯.৫ কোন সার্ভার চালু আছে/নেই চেক করা

```bash
lsof -nP -iTCP:5433 -sTCP:LISTEN   # Postgres
lsof -nP -iTCP:8001 -sTCP:LISTEN   # Embedding service
lsof -nP -iTCP:8000 -sTCP:LISTEN   # মূল API
```

কোনো আউটপুট না এলে বুঝবেন সেই সার্ভিস বন্ধ।

---

### ৯.৬ `brew services list` এ `started` দেখাচ্ছে তবুও connection refused

`brew services` কখনো কখনো server পুরোপুরি রেডি হওয়ার *আগেই* "started" বলে দেয়। সবসময় `pg_isready` দিয়ে নিশ্চিত হয়ে তারপর অ্যাপ চালু করুন (ধারা ৩ দেখুন)।

---

## ১০. দ্রুত রেফারেন্স — একনজরে সব কমান্ড

```bash
# সার্ভার চালু (৩টা আলাদা টার্মিনাল)
brew services start postgresql@18
uv run uvicorn app.embeddings.service:app --port 8001
uv run uvicorn app.main:app --reload --port 8000

# ডাটাবেজ খালি করা
PGPASSWORD=diu psql -h localhost -p 5433 -U diu -d diu_admission \
  -c "TRUNCATE TABLE users, categories, posts, post_files, fail_history, category_post, ai_content, ai_content_vectors RESTART IDENTITY CASCADE;"

# একটা ইউজার বসানো
PGPASSWORD=diu psql -h localhost -p 5433 -U diu -d diu_admission \
  -c "INSERT INTO users (name, email, password, status) VALUES ('Admission Officer (seed)', 'seed@example.invalid', 'NOT-A-REAL-HASH-seed-only', 'active');"

# আটকে থাকা DB connection ভাঙা (deadlock হলে)
PGPASSWORD=diu psql -h localhost -p 5433 -U diu -d diu_admission \
  -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='diu_admission' AND pid <> pg_backend_pid() AND (state = 'idle in transaction' OR wait_event_type='Lock');"

# পোস্ট স্ট্যাটাস চেক
PGPASSWORD=diu psql -h localhost -p 5433 -U diu -d diu_admission \
  -c "SELECT id, title, status, approval_status, verify_status, fail_count FROM posts ORDER BY id;"
```
