🎓 DIU Admission Chatbot (RAG-Powered)

An intelligent, context-aware Admission Chatbot designed for Daffodil International University (DIU). This project leverages Retrieval-Augmented Generation (RAG), combining BM25 Keyword Search and Cosine Vector Similarity Search (Hybrid Search) to deliver highly accurate admission guidance in both English and Bengali.

✨ Features

Hybrid Search Retrieval: Combines BM25 lexical search with dense vector similarity search for precise document context extraction.

Multilingual Support: Handles inquiries seamlessly in English and Bengali.

FastAPI Backend: Lightweight, asynchronous, and high-performance REST APIs.

LangChain Integration: Advanced RAG pipeline for natural language generation and context management.

Ingestion Pipeline: Automatic text chunking, document parsing, and database seeding.

Admin & Chat Endpoints: Separate endpoints for client chat interaction and admin management.

📁 Repository Structure

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


🛠️ Tech Stack

Backend Framework: FastAPI / Python

Orchestration / RAG: LangChain

Search & Retrieval: BM25 Search + Vector Cosine Search (Hybrid)

Database: PostgreSQL / Vector Store

Package Manager: uv / pip

🚀 Getting Started

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

📄 API Documentation

Once the server is running, you can explore the interactive API docs:

Swagger UI: http://localhost:8000/docs

ReDoc: http://localhost:8000/redoc

📝 License

This project is open-source and available under the MIT License.
