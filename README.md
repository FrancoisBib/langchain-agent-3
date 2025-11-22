# LangChain Chatbot

## Overview

**LangChain Chatbot** is a lightweight, extensible example project that demonstrates how to build a conversational AI assistant using **LangChain** with **Retrieval‑Augmented Generation (RAG)**.  The repository showcases best practices for:

- Setting up a LangChain pipeline (LLM → Retriever → Prompt)
- Integrating vector stores (e.g., Chroma, FAISS, Pinecone)
- Managing conversation memory
- Deploying the bot locally or on a cloud platform
- Writing tests and contributing new features

The goal is to provide developers with a clear, production‑ready reference they can clone, adapt, and extend.

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Bot](#running-the-bot)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Contribution Guide](#contribution-guide)
- [License](#license)

---

## Features

- **LangChain Core**: Utilises `langchain` to orchestrate LLM calls, memory, and retrieval.
- **RAG Architecture**: Combines a vector store retriever with a language model to answer queries grounded in external documents.
- **Modular Design**: Plug‑and‑play components for LLMs, embeddings, and vector stores.
- **CLI & API**: Interact via a simple command‑line interface or expose a FastAPI endpoint.
- **Docker Support**: Ready-to‑run container with all dependencies isolated.
- **Testing Suite**: Unit and integration tests using `pytest` and `pytest‑asyncio`.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python      | >=3.9   |
| pip         | latest  |
| Docker (optional) | >=20.10 |
| OpenAI API key (or compatible LLM endpoint) |
| Optional: Pinecone/Weaviate credentials for remote vector stores |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-org/langchain-chatbot.git
cd langchain-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Install the package in editable mode

```bash
pip install -e .
```

---

## Configuration

The bot reads its configuration from a **`.env`** file located at the project root.  Example template:

```dotenv
# .env
# LLM configuration
OPENAI_API_KEY=sk-****************
LLM_MODEL=gpt-3.5-turbo
TEMPERATURE=0.7

# Embedding model (OpenAI or local)
EMBEDDING_MODEL=text-embedding-ada-002

# Vector store selection (chroma, faiss, pinecone, weaviate)
VECTOR_STORE=chroma
CHROMA_PERSIST_DIR=./data/chroma

# Retrieval parameters
TOP_K=4

# FastAPI settings (if using the API server)
HOST=0.0.0.0
PORT=8000
```

> **Tip**: Do not commit the real `.env` file. Add it to `.gitignore` (already present).

---

## Running the Bot

### CLI mode

```bash
python -m langchain_chatbot.cli
```

You will be dropped into an interactive prompt where you can ask questions.  The bot will retrieve relevant chunks from the vector store and generate a response using the configured LLM.

### API mode (FastAPI)

```bash
uvicorn langchain_chatbot.api:app --host $HOST --port $PORT
```

The API exposes a single endpoint:

- `POST /chat` – body `{ "message": "Your question" }`
- Returns `{ "answer": "Generated response", "sources": ["doc1.txt", ...] }`

### Docker

```bash
docker build -t langchain-chatbot .
docker run -p 8000:8000 --env-file .env langchain-chatbot
```

---

## Project Structure

```
langchain-chatbot/
│
├── src/                     # Source package
│   ├── __init__.py
│   ├── chatbot.py           # Core LangChain pipeline (LLM + Retriever)
│   ├── memory.py            # Conversation memory utilities
│   ├── retriever.py         # Vector store abstraction
│   ├── config.py            # Pydantic settings loader
│   ├── cli.py               # Command‑line interface
│   └── api.py               # FastAPI server
│
├── data/                    # Persistent vector store files (if using Chroma/FAISS)
│
├── tests/                   # Test suite
│   ├── unit/
│   └── integration/
│
├── .env.example             # Example environment file
├── requirements.txt
├── pyproject.toml           # Build metadata (editable install)
├── Dockerfile
└── README.md                # <‑‑ you are reading it!
```

---

## Testing

Run the full test suite with:

```bash
pytest -v
```

### Adding new tests

- **Unit tests** go under `tests/unit/` and should mock external services (LLM, vector store).
- **Integration tests** under `tests/integration/` may spin up a temporary in‑memory vector store and use a real LLM key.

All tests must pass before opening a pull request.

---

## Contribution Guide

We welcome contributions! Follow these steps:

1. **Fork** the repository and **clone** your fork.
2. Create a **feature branch**:
   ```bash
   git checkout -b feat/your-feature-name
   ```
3. Make your changes, ensuring the code follows the project's **PEP‑8** style and passes `pytest`.
4. Update the documentation (README, docstrings) as needed.
5. Commit with a clear message and push:
   ```bash
   git push origin feat/your-feature-name
   ```
6. Open a **Pull Request** against `main`.  Include a concise description of the change and reference any related issue.

### Code of Conduct

Please adhere to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).  Harassment or exclusionary behavior will not be tolerated.

---

## License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## Acknowledgments

- The LangChain team for the powerful framework.
- OpenAI for the API that powers the LLM component.
- The open‑source community for vector‑store implementations.

---

*Happy coding!*