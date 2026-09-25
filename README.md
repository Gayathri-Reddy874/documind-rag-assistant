# 📚 DocuMind - Enterprise Document Intelligence Assistant

<p>
  <a href="https://github.com/Gayathri-Reddy874/documind-rag-assistant/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Gayathri-Reddy874/documind-rag-assistant/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-blue.svg">
  <a href="https://streamlit.io/"><img alt="Streamlit" src="https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg"></a>
  <a href="https://groq.com/"><img alt="Groq" src="https://img.shields.io/badge/LLM-Groq%20LPU-F55036.svg"></a>
  <a href="https://github.com/Gayathri-Reddy874/documind-rag-assistant/blob/main/LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/Gayathri-Reddy874/documind-rag-assistant">
</p>

A production-structured **Retrieval-Augmented Generation (RAG)** chatbot that lets you upload documents (PDF, DOCX, TXT, CSV, Markdown) and ask natural-language questions about them - grounded, cited answers powered by **Groq's LPU inference** for near-instant responses.

Built with a clean, layered architecture (config → ingestion → vector store → LLM chain → UI) so each piece is independently testable, swappable, and container-ready.

---

## 📸 Screenshots

| Configuration & Upload | Document Indexed |
|---|---|
| ![Groq API key entry and file upload panel](<Screenshots/GroqKey & File_upload.png>) | ![Document successfully indexed confirmation](Screenshots/Indexed.png) |

| Chat Preview | Retrieval with Sources |
|---|---|
| ![Chat interface preview](Screenshots/Preview.png) | ![Answer with retrieved source chunks](Screenshots/Retrieval.png) |

---

## ✨ Features

- **Multi-format ingestion** - PDF, DOCX, TXT, Markdown, and CSV, with per-file validation (type, size, empty-content checks).
- **Conversational memory** - follow-up questions are automatically reformulated into standalone queries using chat history (no more "what did you mean by that?").
- **Source-grounded answers** - every response links back to the exact document chunks it was generated from, shown inline in the UI.
- **Groq-powered inference** - swap between `openai/gpt-oss-120b` and `openai/gpt-oss-20b` from the sidebar (see [Groq's model list](https://console.groq.com/docs/models) for what's currently available on your plan).
- **Local, private vector search** — FAISS + Sentence-Transformers embeddings run entirely on your machine; documents never leave your environment except for the LLM call itself.
- **Persistence-ready** — vector indices can be saved/loaded per session for reuse across restarts.
- **Enterprise-grade error handling** — typed, domain-specific exceptions surfaced as clear UI messages instead of stack traces.
- **Fully tested & CI'd** — unit tests with mocked dependencies, linting (ruff + black), and a GitHub Actions pipeline that also builds the Docker image.
- **Container-first** — multi-stage `Dockerfile`, non-root user, health checks, and a ready-to-go `docker-compose.yml`.

---

## 🏗️ Architecture

```mermaid
flowchart LR
    U[User] -->|Uploads file| UI[Streamlit UI]
    UI --> DP[DocumentProcessor]
    DP -->|validate, load, chunk| VS[VectorStoreManager]
    VS -->|FAISS + embeddings| IDX[(Vector Index)]

    U -->|Asks question| UI
    UI --> RAG[RAGChatbot]
    RAG -->|reformulate w/ history| LLM1[Groq LLM]
    RAG -->|similarity search| IDX
    IDX -->|top-k chunks| RAG
    RAG -->|context + question| LLM2[Groq LLM]
    LLM2 -->|grounded answer + sources| UI
```

### Project structure

```
documind-rag-assistant/
├── app.py                      # Streamlit UI (entry point)
├── src/
│   ├── config.py                # Pydantic settings, env-driven
│   ├── logging_config.py        # Centralized logging setup
│   ├── exceptions.py            # Domain-specific exception types
│   ├── document_processor.py    # Load, validate, chunk documents
│   ├── vector_store.py          # FAISS index build/persist/query
│   └── rag_chain.py             # LCEL conversational RAG chain (Groq)
├── tests/
│   ├── test_document_processor.py
│   └── test_vector_store.py
├── Screenshots/                 # README screenshots
├── .github/workflows/ci.yml     # Lint, test, Docker build
├── Dockerfile                   # Multi-stage, non-root, healthcheck
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml               # ruff / black / pytest / mypy config
├── .env.example
└── LICENSE
```

---

## 🚀 Getting started

### 1. Clone & set up a virtual environment

```bash
git clone https://github.com/Gayathri-Reddy874/documind-rag-assistant.git
cd documind-rag-assistant
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY (free key at https://console.groq.com)
```

### 4. Run the app

```bash
streamlit run app.py
```

Open **http://localhost:8501**, upload a document from the sidebar, click **Process documents**, and start asking questions.

---

## 🐳 Run with Docker

```bash
docker compose up --build
```

The app will be available at **http://localhost:8501**. Vector indices persist to `./data/vector_store` on the host via a mounted volume.

---

## 🧪 Testing & code quality

```bash
pip install -r requirements-dev.txt

ruff check src tests app.py      # lint
black --check src tests app.py   # formatting
pytest --cov=src                 # unit tests + coverage
```

All three run automatically in CI on every push and pull request to `main`.

---

## ⚙️ Configuration reference

All settings are environment-driven (see `.env.example`) and validated at startup via Pydantic:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Your Groq API key (required) |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Groq-hosted generation model — check [console.groq.com/docs/models](https://console.groq.com/docs/models) if you hit `model_not_found` |
| `LLM_TEMPERATURE` | `0.2` | Sampling temperature |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Text splitting parameters |
| `RETRIEVER_TOP_K` | `4` | Number of chunks retrieved per query |
| `MAX_UPLOAD_MB` | `25` | Per-file upload size limit |

---

## 🗺️ Roadmap

- [ ] Swap FAISS for a managed vector DB (Pinecone / Qdrant) behind the same `VectorStoreManager` interface
- [ ] Add streaming token-by-token responses in the UI
- [ ] Multi-user auth and per-user document namespaces
- [ ] Evaluation harness (RAGAS) for answer faithfulness and retrieval precision

---

## 🤝 Contributing

Issues and pull requests are welcome. Please run `ruff`, `black`, and `pytest` locally before opening a PR — the CI pipeline enforces all three.

---

## 👤 Author

**Mallareddygari Gayathri**

[![GitHub](https://img.shields.io/badge/GitHub-Gayathri--Reddy874-181717?logo=github)](https://github.com/Gayathri-Reddy874)

---

## 📄 License

Released under the [MIT License](LICENSE) - free to use, modify, and distribute with attribution.
