# LangChain Chatbot with Retrieval‑Augmented Generation (RAG)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

A minimal, production‑ready example of a **chatbot** built on top of **LangChain** that leverages **Retrieval‑Augmented Generation (RAG)**. The repository demonstrates how to:

- Connect a large language model (LLM) to a vector store for document retrieval.
- Build a LangChain **ConversationalRetrievalChain** that keeps context across turns.
- Deploy the bot locally (CLI) or as a simple FastAPI endpoint.
- Run end‑to‑end tests and contribute improvements.

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Bot](#running-the-bot)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **LangChain** integration with a configurable LLM (OpenAI, Anthropic, Ollama, …).
- **RAG** pipeline using **FAISS** (or any other `VectorStore` compatible with LangChain).
- Stateful chat with a **ConversationBufferMemory**.
- Simple CLI for quick prototyping.
- Optional FastAPI server for HTTP‑based interaction.
- Type‑annotated, PEP‑8 compliant code base.

---

## Prerequisites

- Python **3.9** or newer.
- An API key for the LLM you intend to use (e.g., `OPENAI_API_KEY`).
- Optional: `git`, `make` (for convenience scripts).

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/langchain-chatbot.git
cd langchain-chatbot

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # on Windows use `.venv\Scripts\activate`

# Install the core dependencies
pip install -r requirements.txt
```

> **Tip**: The `requirements.txt` pins versions that are known to work together. If you need the latest LangChain features, upgrade with `pip install -U langchain` and adjust the code accordingly.

---

## Configuration

All configurable values are read from environment variables. Create a `.env` file at the project root (it is ignored by Git) and populate it with the keys you need:

```dotenv
# LLM configuration (choose one)
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=...
# OLLAMA_BASE_URL=http://localhost:11434

# Vector store settings – FAISS uses a local index file
FAISS_INDEX_PATH=./data/faiss_index

# Optional FastAPI settings
HOST=0.0.0.0
PORT=8000
```

The application loads the `.env` file automatically via **python‑dotenv**.

---

## Running the Bot

### 1️⃣ CLI Mode (quick test)

```bash
python -m chatbot.cli
```

You will be prompted for a question; the bot will retrieve relevant chunks, generate a response, and keep the conversation context.

### 2️⃣ FastAPI Server

```bash
uvicorn chatbot.api:app --host $HOST --port $PORT
```

The server exposes a single endpoint:

- `POST /chat` – body `{ "question": "..." }`
- Returns `{ "answer": "...", "source_documents": [...] }`

You can test it with `curl` or any HTTP client:

```bash
curl -X POST http://localhost:8000/chat \
     -H 'Content-Type: application/json' \
     -d '{"question": "What is Retrieval‑Augmented Generation?"}'
```

---

## Project Structure

```
langchain-chatbot/
├─ chatbot/                 # Core package
│   ├─ __init__.py
│   ├─ config.py            # Environment handling
│   ├─ retrieval.py         # Vector store & retriever setup
│   ├─ chain.py             # ConversationalRetrievalChain builder
│   ├─ cli.py               # Simple command‑line interface
│   └─ api.py               # FastAPI endpoint
├─ data/                    # Persisted FAISS index, sample docs
├─ tests/                   # Unit & integration tests
│   └─ test_chatbot.py
├─ .env.example             # Template for environment variables
├─ requirements.txt
├─ README.md                # ← you are reading this file
└─ pyproject.toml           # (optional) build metadata
```

---

## Testing

The repository includes a minimal test suite based on **pytest**.

```bash
pytest -q
```

Tests cover:
- Vector store loading
- Retrieval logic
- End‑to‑end conversation flow (mocked LLM)

Feel free to add more tests for edge cases or new features.

---

## Contributing

Contributions are welcome! Follow these steps:

1. **Fork** the repository and **clone** your fork.
2. Create a **feature branch**: `git checkout -b feature/your‑feature`.
3. Install the development dependencies:
   ```bash
   pip install -r dev-requirements.txt
   ```
4. Make your changes, ensuring that the code passes `pytest` and complies with **flake8**/`black` formatting.
5. Open a **Pull Request** targeting the `main` branch. Provide a clear description of the change and reference any related issues.

Please read the full [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on coding standards, commit messages, and the review process.

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- **LangChain** – the framework that makes composable LLM applications easy.
- The LangChain community for examples and best‑practice patterns.
- The open‑source contributors of FAISS, FastAPI, and related tooling.
