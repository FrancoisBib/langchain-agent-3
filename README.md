# LangChain Chatbot with Retrieval‑Augmented Generation (RAG)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-brightgreen.svg)](https://www.python.org/downloads/)

A minimal yet extensible **Chatbot** built on **[LangChain](https://github.com/langchain-ai/langchain)** that demonstrates **Retrieval‑Augmented Generation (RAG)**.  The bot can:

- **Answer questions** using a vector store of your own documents.
- **Maintain conversational context** with LangChain memory utilities.
- **Be extended** with custom LLMs, retrievers, or UI front‑ends.

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Run the demo](#run-the-demo)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **LangChain integration** – uses `ChatOpenAI`, `ConversationChain`, and memory modules.
- **RAG pipeline** – combines a vector store (FAISS by default) with a retriever to augment LLM responses with relevant document excerpts.
- **Modular design** – swap out LLMs, embeddings, or vector stores with a single line change.
- **CLI & API ready** – a minimal command‑line interface (`python -m chatbot`) and a FastAPI wrapper (`app.py`).
- **Docker support** – build and run the whole stack in containers.

---

## Quick Start

### Prerequisites

- Python **3.9+**
- An OpenAI API key (or any compatible LLM endpoint). Set it in the environment variable `OPENAI_API_KEY`.
- (Optional) `git` and `docker` if you prefer containerised execution.

### Installation

```bash
# Clone the repository
git clone https://github.com/your‑org/langchain-chatbot.git
cd langchain-chatbot

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Tip:** The `requirements.txt` pins `langchain`, `openai`, `faiss-cpu`, and `fastapi`. Adjust versions as needed.

### Run the demo

1. **Prepare a document corpus** – place plain‑text (`.txt`) files inside the `data/` directory. The demo ships with a few sample PDFs that are automatically converted.
2. **Index the documents**:
   ```bash
   python -m chatbot.index
   ```
   This creates a `faiss_index` folder containing the vector store.
3. **Start the chatbot** (CLI mode):
   ```bash
   python -m chatbot.chat
   ```
   You can now type questions and see RAG‑augmented answers.

   Or launch the FastAPI server:
   ```bash
   uvicorn app:app --reload
   ```
   Open `http://127.0.0.1:8000/docs` for an interactive Swagger UI.

---

## Project Structure

```
langchain-chatbot/
├─ chatbot/                 # Core package
│   ├─ __init__.py
│   ├─ config.py            # Pydantic settings (LLM, embeddings, etc.)
│   ├─ retriever.py         # Vector store + retrieval logic
│   ├─ chain.py             # LangChain chain definition
│   ├─ chat.py              # CLI entry point
│   └─ index.py             # Document ingestion & FAISS indexing
├─ data/                    # Sample documents (txt, pdf, md…) – add your own here
├─ tests/                   # Unit & integration tests
│   └─ test_chatbot.py
├─ app.py                   # FastAPI wrapper exposing `/chat` endpoint
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
└─ README.md                # <‑‑ you are reading this file
```

---

## How It Works

1. **Embedding creation** – `OpenAIEmbeddings` (or any `Embeddings` implementation) turns each document chunk into a dense vector.
2. **Vector store** – FAISS stores the vectors for fast similarity search.
3. **Retriever** – `FAISS`'s `as_retriever()` returns the top‑k most relevant chunks for a query.
4. **RAG chain** – `ConversationalRetrievalChain` merges the retrieved context with the user prompt and sends it to the LLM.
5. **Memory** – `ConversationBufferMemory` (or a custom memory) keeps the dialogue history, enabling follow‑up questions.

---

## Configuration

All runtime options live in `chatbot/config.py` and are loaded via **pydantic** `BaseSettings`. Example environment variables:

```dotenv
# .env (optional)
OPENAI_API_KEY=sk-xxxxxx
EMBEDDING_MODEL=text-embedding-ada-002
LLM_MODEL=gpt-3.5-turbo
TOP_K=4                # number of retrieved chunks per query
MAX_TOKENS=1024        # LLM response length
```

You can also override settings programmatically:

```python
from chatbot.config import Settings
settings = Settings(openai_api_key="sk-...", top_k=6)
```

---

## Testing

The repository includes a small test suite using **pytest**.

```bash
pytest -q
```

Key tests cover:
- Document indexing and vector store persistence.
- Retrieval correctness (top‑k relevance).
- End‑to‑end conversation flow with mocked LLM responses.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository** and create a feature branch.
2. **Install the development dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```
3. **Run the test suite** before committing.
4. **Write clear commit messages** – the first line should be a concise summary (≤50 chars).
5. **Open a Pull Request** targeting the `main` branch.

### Code style

- Use **black** for formatting (`black .`).
- Lint with **ruff** (`ruff check .`).
- Type‑check with **mypy** (`mypy chatbot`).

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- The **LangChain** community for the powerful abstractions.
- **FAISS** for efficient similarity search.
- OpenAI for the underlying language models.

---

*Happy coding! 🚀*