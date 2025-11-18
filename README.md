# LangChain Chatbot with Retrieval‑Augmented Generation (RAG)

## Overview

This repository provides a **minimal, production‑ready chatbot** built on top of **[LangChain](https://github.com/langchain-ai/langchain)**.  It demonstrates how to combine:

- **Large Language Models (LLMs)** for natural‑language generation
- **Vector stores** for semantic retrieval
- **Retrieval‑Augmented Generation (RAG)** to ground responses in external knowledge

The bot is packaged as a simple FastAPI service, but the core logic is framework‑agnostic and can be reused in other environments (CLI, Streamlit, etc.).

---

## Table of Contents

- [Features](#features)
- [Architecture Diagram](#architecture-diagram)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the API](#running-the-api)
- [Usage](#usage)
  - [API Endpoints](#api-endpoints)
  - [Example Requests](#example-requests)
- [Configuration](#configuration)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **RAG pipeline** using LangChain `Retriever` + `LLMChain`
- **Pluggable vector stores** (currently supports **FAISS**, **Chroma**, **Pinecone**)
- **Prompt templating** for consistent system messages
- **Streaming responses** via FastAPI `EventSourceResponse`
- **Docker support** for reproducible deployments
- **Extensible architecture** – swap LLMs, retrievers, or storage back‑ends with a single line change

---

## Architecture Diagram

```
+----------------+      +----------------+      +-------------------+
|   User Input   | ---> |   FastAPI      | ---> | LangChain RAG     |
| (HTTP/WS)      |      |   Endpoint     |      |  ├─ Retriever     |
+----------------+      +----------------+      |  └─ LLMChain       |
                                                +-------------------+
```

1. **FastAPI** receives the request and forwards the user query to the LangChain RAG pipeline.
2. The **Retriever** searches a vector store for the most relevant documents.
3. The **LLMChain** combines the retrieved context with a system prompt and generates a response.
4. The response is streamed back to the client.

---

## Getting Started

### Prerequisites

- Python **3.10** or newer
- An OpenAI API key (or another LLM provider supported by LangChain)
- Optional: Docker & Docker‑Compose if you prefer containerised execution

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/langchain-chatbot.git
cd langchain-chatbot

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # on Windows use `.venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

If you want to use **FAISS** (the default local vector store), ensure you have the required compiled libraries:

```bash
pip install "faiss-cpu>=1.7.4"
```

### Running the API

```bash
# Export your OpenAI key (or set it in a .env file)
export OPENAI_API_KEY='sk-...'

# Start the FastAPI server
uvicorn app.main:app --reload
```

The service will be available at `http://127.0.0.1:8000`.

---

## Usage

### API Endpoints

| Method | Path               | Description                              |
|--------|--------------------|------------------------------------------|
| POST   | `/chat`            | Send a user message and receive a reply  |
| GET    | `/health`          | Simple health‑check endpoint              |

### Example Request (cURL)

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Explain the difference between supervised and unsupervised learning."}'
```

The response is streamed JSON lines of the form:

```json
{"content": "Supervised learning ..."}
```

---

## Configuration

All configurable values are read from environment variables (see `.env.example`).  Common options include:

- `OPENAI_API_KEY` – your OpenAI key
- `LLM_MODEL` – e.g., `gpt-3.5-turbo` (default)
- `VECTOR_STORE` – `faiss`, `chroma`, or `pinecone`
- `EMBEDDING_MODEL` – e.g., `text-embedding-ada-002`
- `TOP_K_RETRIEVAL` – number of documents to fetch (default: `4`)

You can override any of these at runtime or by editing the `.env` file.

---

## Testing

The repository includes a small test suite based on **pytest**.

```bash
pip install pytest
pytest
```

Tests cover:
- Retriever integration
- Prompt templating
- API response shape

---

## Contributing

Contributions are welcome!  Follow these steps:

1. **Fork** the repository.
2. Create a feature branch: `git checkout -b feature/awesome-feature`.
3. Make your changes and ensure the test suite passes.
4. Submit a **Pull Request** with a clear description of the change.
5. Follow the existing code style (Black, isort, flake8).

Please read the [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.
