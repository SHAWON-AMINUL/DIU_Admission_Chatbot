"""Single source of truth for configuration.

Everything the ingest pipeline and the embedding service need to agree on
lives here so the two sides cannot drift apart.
"""

import os

from dotenv import load_dotenv

load_dotenv()


# --- Database -------------------------------------------------------------
# Port 5433: the machine already runs an EnterpriseDB PostgreSQL on 5432 and
# we deliberately leave that installation alone.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://diu:diu@localhost:5433/diu_admission",
)


# --- Embedding model ------------------------------------------------------
# BGE-M3: multilingual (100+ languages incl. Bangla), 1024 dimensions,
# symmetric (no query/passage prefix ritual to get wrong between the
# ingest side and the query side).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# Written verbatim into ai_content_vectors.model_name on every row. The
# dimension is part of the model's identity: a future re-embed at a different
# width must be distinguishable from this one.
MODEL_NAME_TAG = os.getenv("MODEL_NAME_TAG", "bge-m3@1024")

EMBEDDING_DIM = 1024

# The embedding microservice both the ingest pipeline and the query side call.
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://127.0.0.1:8001")

# "mps" uses the M1 Pro GPU. Falls back automatically if unavailable.
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")


# --- Chunking -------------------------------------------------------------
# Token counts are BGE-M3 subword tokens, measured with the model's own
# tokenizer. Bangla expands to far more subword tokens per character than
# English, so a word count or tiktoken would size Bangla chunks wrongly.
CHUNK_TARGET_TOKENS = int(os.getenv("CHUNK_TARGET_TOKENS", "300"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "60"))

# BGE-M3 accepts 8192 tokens; we stay far below that on purpose. Admission
# content is short and factual, and oversized chunks dilute the embedding.
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "512"))


# --- Retrieval ------------------------------------------------------------
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
# Reciprocal Rank Fusion damping constant. 60 is the value from the original
# RRF paper and is what almost every hybrid-search implementation uses.
RRF_K = int(os.getenv("RRF_K", "60"))


# --- Generation -----------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

# gemma3:4b, chosen by probe rather than reputation. The previously-installed
# milkey/Kalomaze-Qwen3-16B-A3B cannot generate Bangla at all -- Ollama returns
# HTTP 500 the moment a Bangla reply is requested. gemma3 is trained on 140+
# languages and answers Bangla correctly.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")

# Near-zero: this bot restates admission facts, it does not write prose.
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))

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
