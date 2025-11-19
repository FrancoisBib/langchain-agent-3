# LangChain Chatbot with Retrieval‑Augmented Generation (RAG)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-brightgreen.svg)](https://www.python.org/downloads/)

A minimal yet extensible reference implementation of a **chatbot powered by LangChain** that combines **large language models (LLMs)** with **retrieval‑augmented generation (RAG)**.  The project demonstrates how to:

- Load and index documents with a vector store (FAISS, Chroma, …).
- Retrieve relevant chunks at query time.
- Feed retrieved context to an LLM (OpenAI, Anthropic, Cohere, …) to generate informed answers.
- Expose the bot through a simple CLI and a FastAPI web endpoint.

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the demo](#running-the-demo)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Extending the Bot](#extending-the-bot)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **LangChain core**: chain together LLMs, prompts, and retrievers.
- **RAG pipeline**: document ingestion → embedding → vector store → similarity search.
- **Modular design**: swap out LLM providers, vector stores, or document loaders without touching the main logic.
- **CLI & HTTP API**: interact with the bot locally or integrate it into other services.
- **Typed settings**: Pydantic‑based configuration for reproducibility.
- **Unit tests** covering the retrieval and generation steps.

---

## Quick Start

### Prerequisites

- Python **3.9+**
- An OpenAI API key (or any other LLM provider key you plan to use)
- `git` (to clone the repository)

### Installation

```bash
# Clone the repository
git clone https://github.com/your‑org/langchain-chatbot.git
cd langchain-chatbot

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# Install the package in editable mode with all extras
pip install -e .[dev]
```

The `dev` extra installs testing utilities (`pytest`), linting (`ruff`), and optional vector‑store back‑ends.

### Running the demo

1. **Add your API key** – create a `.env` file at the project root:

   ```dotenv
   OPENAI_API_KEY=sk-********************************
   ```

2. **Load some documents** – the repository ships with a small `data/` folder.  You can also point the script to any directory of `.txt`, `.pdf`, or `.md` files.

   ```bash
   python scripts/ingest.py --source data/
   ```

   This will:
   - Split documents into chunks.
   - Compute embeddings (default: `text-embedding-ada-002`).
   - Persist a FAISS index in `vector_store/`.

3. **Start the chatbot** – either via CLI or the API:

   ```bash
   # CLI mode
   python -m langchain_chatbot.cli
   ```

   ```bash
   # FastAPI server (default on http://127.0.0.1:8000)
   uvicorn langchain_chatbot.api:app --reload
   ```

   Example interaction (CLI):
   ```text
   > Hello! How can I help you?
   User: What is the purpose of this repository?
   Bot: This repository demonstrates …
   ```

---

## Project Structure

```
langchain-chatbot/
├─ langchain_chatbot/          # Core package
│   ├─ __init__.py
│   ├─ config.py               # Pydantic settings
│   ├─ loaders.py              # Document loaders (txt, pdf, md …)
│   ├─ embeddings.py           # Embedding model wrapper
│   ├─ vector_store.py         # FAISS / Chroma helper
│   ├─ retriever.py            # Retrieval chain
│   ├─ llm.py                  # LLM wrapper (OpenAI, Anthropic, …)
│   ├─ chatbot.py              # High‑level RAG chain
│   ├─ cli.py                  # Command‑line interface
│   └─ api.py                  # FastAPI endpoint
├─ scripts/                    # Utility scripts
│   └─ ingest.py               # Document ingestion & index creation
├─ tests/                      # Unit and integration tests
│   └─ test_chatbot.py
├─ data/                       # Sample documents (optional)
├─ .env.example                # Template for environment variables
├─ pyproject.toml              # Build system & dependencies
├─ README.md                   # ← you are here
└─ LICENSE
```

---

## Configuration

All runtime options are defined in `langchain_chatbot/config.py` and can be overridden via environment variables or a `.env` file.

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI secret key | – |
| `EMBEDDING_MODEL` | Embedding model name (OpenAI) | `text-embedding-ada-002` |
| `LLM_MODEL` | LLM model name (OpenAI) | `gpt-3.5-turbo` |
| `VECTOR_STORE` | Vector store backend (`faiss`, `chroma`) | `faiss` |
| `CHUNK_SIZE` | Number of characters per text chunk | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |

You can also provide a custom `config.yaml` and load it with `python -m langchain_chatbot.cli --config path/to/config.yaml`.

---

## Extending the Bot

The code is deliberately modular:

1. **Swap the LLM** – implement a new class inheriting from `BaseLLM` in `llm.py` and register it in `config.py`.
2. **Change the vector store** – add a wrapper in `vector_store.py` (e.g., Pinecone, Weaviate) and expose it via the `VECTOR_STORE` setting.
3. **Add new loaders** – extend `loaders.py` with a function that returns a list of `Document` objects for your file type.
4. **Custom prompts** – edit `chatbot.py` where the `PromptTemplate` lives; you can store templates in `templates/` for easier versioning.

---

## Testing

```bash
pytest -q
```

The test suite includes:
- Unit tests for the retriever and LLM wrappers.
- An integration test that runs a short end‑to‑end RAG query using a tiny in‑memory vector store.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository** and clone your fork.
2. **Create a feature branch** (`git checkout -b feat/awesome-feature`).
3. **Install the dev dependencies** (`pip install -e .[dev]`).
4. **Write tests** for any new functionality.
5. **Run the linter** (`ruff check .`).
6. **Submit a Pull Request** with a clear description of the change.

See `CONTRIBUTING.md` for detailed guidelines on code style, commit messages, and release process.

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.
