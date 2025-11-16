# LangChain Chatbot with Retrieval-Augmented Generation (RAG)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Running the Bot](#running-the-bot)
- [Testing](#testing)
- [Contribution Guide](#contribution-guide)
- [License](#license)

---

## Overview

`langchain-chatbot` is a modular, extensible chatbot built on **LangChain** that demonstrates **Retrieval‑Augmented Generation (RAG)**. It combines a large language model (LLM) with a vector store to retrieve relevant documents at inference time, enabling the bot to answer questions with up‑to‑date, factual context.

The project showcases:
- Integration of LangChain agents, chains and memory.
- Use of popular vector stores (FAISS, Chroma, Pinecone) via a unified interface.
- A clean, testable codebase that follows the **Clean Architecture** principles.
- Docker support for reproducible development and deployment.

---

## Features

- **RAG pipeline**: Retrieve relevant chunks from a document corpus and feed them to the LLM.
- **Multi‑model support**: Switch between OpenAI, Anthropic, Cohere, or any OpenAI‑compatible endpoint.
- **Configurable retrievers**: BM25, semantic similarity, hybrid retrieval.
- **Prompt templates**: Ready‑made system prompts for Q&A, summarisation, and chat.
- **Streaming responses**: Optional token‑by‑token streaming for a smoother UI.
- **Extensible**: Add new document loaders, vector stores, or post‑processing steps with minimal code changes.
- **Dockerised**: Run the whole stack (app + vector store) with a single `docker compose up`.

---

## Architecture

```
src/
├─ config/          # Pydantic settings (API keys, model params)
├─ loaders/         # Document loaders (pdf, txt, html, markdown)
├─ embeddings/      # Embedding models (OpenAI, HuggingFace)
├─ vectorstores/    # FAISS, Chroma, Pinecone wrappers
├─ retrievers/      # Retrieval logic (similarity, hybrid)
├─ chains/          # LangChain chains (RAGChain, ChatChain)
├─ api/             # FastAPI entry point (optional UI)
├─ tests/           # Pytest suite
└─ main.py          # CLI entry point
```

The core **RAGChain** orchestrates the flow:
1. **Load** documents → split into chunks.
2. **Embed** chunks → store in a vector store.
3. **Retrieve** top‑k relevant chunks for a user query.
4. **Generate** answer with the LLM using a prompt that includes retrieved context.
5. **Return** the answer (optionally with source citations).

---

## Installation

### Prerequisites
- Python **3.10** or newer
- `git`
- (Optional) Docker & Docker‑Compose if you prefer containerised execution

### Clone the repository
```bash
git clone https://github.com/your-org/langchain-chatbot.git
cd langchain-chatbot
```

### Create a virtual environment and install dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Environment variables
Create a `.env` file at the project root with the required keys:
```dotenv
# LLM provider (OPENAI, ANTHROPIC, COHERE, ...)
LLM_PROVIDER=OPENAI
OPENAI_API_KEY=sk-...
# Embedding model (default: text-embedding-ada-002)
EMBEDDING_MODEL=text-embedding-ada-002
# Vector store configuration (FAISS is local, Pinecone requires API key)
VECTORSTORE=FAISS
# Optional Pinecone configuration
PINECONE_API_KEY=...
PINECONE_ENV=us-east1-gcp
```

---

## Quick Start

### 1️⃣ Index your knowledge base
```bash
python -m src.indexer --source data/knowledge/ --store faiss
```
- `--source` points to a folder containing PDFs, TXT, Markdown, or HTML files.
- `--store` selects the vector store (`faiss`, `chroma`, `pinecone`).

### 2️⃣ Launch the chatbot (CLI)
```bash
python -m src.main
```
You will be prompted for a question; the bot will retrieve relevant passages and generate a response.

### 3️⃣ Launch the optional FastAPI UI
```bash
uvicorn src.api:app --reload
```
Open `http://localhost:8000/docs` to explore the OpenAPI UI or `http://localhost:8000` for a minimal web front‑end.

---

## Configuration

All runtime settings are defined via **Pydantic** models in `src/config/settings.py`. You can override defaults by editing the `.env` file or passing environment variables.

Key sections:
- **LLMSettings** – model name, temperature, max tokens.
- **EmbeddingSettings** – model name, dimension.
- **RetrieverSettings** – `k` (number of retrieved documents), `search_type` (similarity, mmr).
- **VectorStoreSettings** – type and connection details.

---

## Running the Bot

### Command‑line interface
```bash
python -m src.main \
    --question "What are the benefits of RAG?" \
    --top_k 5
```
- `--question` can be omitted to enter an interactive REPL.
- `--top_k` controls how many chunks are retrieved.

### Docker Compose (recommended for production)
```bash
docker compose up -d
```
The compose file starts:
- The chatbot service (exposes port 8000 for the API).
- A Chroma vector store container (or any other store you configure).

---

## Testing

Run the test suite with:
```bash
pytest -q
```
The repository includes unit tests for:
- Document loaders
- Embedding wrappers
- Retriever logic
- End‑to‑end RAG chain (mocked LLM)

---

## Contribution Guide

We welcome contributions! Please follow these steps:
1. **Fork** the repository.
2. **Create a branch** for your feature or bugfix:
   ```bash
   git checkout -b feat/your-feature-name
   ```
3. **Write tests** for new functionality.
4. **Run the full test suite** to ensure nothing breaks.
5. **Submit a Pull Request** with a clear description of the change.

### Code style
- Use **Black** for formatting (`black .`).
- Lint with **ruff** (`ruff check .`).
- Type‑check with **mypy** (`mypy src`).

### Documentation updates
If you add new public functions or classes, update the docstrings and the relevant sections of this README.

---

## License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.

---

## Acknowledgements

- **LangChain** – the core framework that powers the retrieval and generation pipelines.
- **FAISS / Chroma / Pinecone** – vector store back‑ends used for similarity search.
- The open‑source community for providing excellent document loaders and embedding models.
