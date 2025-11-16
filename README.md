# LangChain Chatbot

**LangChain‑Chatbot** is a minimal yet extensible example of building a Retrieval‑Augmented Generation (RAG) chatbot with **[LangChain](https://python.langchain.com/)**, **OpenAI** (or any compatible LLM), and a vector store. The repository demonstrates how to:

- Load documents (Markdown, PDF, txt, etc.)
- Split them into chunks and embed them with a vector store (FAISS, Chroma, etc.)
- Retrieve relevant passages at query time
- Combine retrieved context with a Large Language Model to generate accurate answers
- Deploy the bot locally via a simple CLI or expose it through a FastAPI endpoint

The goal is to provide a clear, well‑documented starting point for developers who want to experiment with RAG‑powered conversational agents or contribute enhancements.

---

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Running the Chatbot](#running-the-chatbot)
   - [CLI mode](#cli-mode)
   - [FastAPI server](#fastapi-server)
6. [Project Structure](#project-structure)
7. [How RAG Works in This Project](#how-rag-works-in-this-project)
8. [Testing](#testing)
9. [Contributing](#contributing)
10. [License](#license)

---

## Features

- **Document ingestion** – Supports PDFs, Markdown, plain‑text, and CSV files.
- **Chunking & embedding** – Uses LangChain `RecursiveCharacterTextSplitter` and any OpenAI‑compatible embedding model.
- **Vector store abstraction** – Plug‑and‑play with FAISS (default), Chroma, or Pinecone.
- **RAG pipeline** – Retrieval with similarity search → LLM prompt templating → generation.
- **CLI & API** – Interact via terminal or HTTP (FastAPI).
- **Typed config** – `pydantic`‑based settings for easy environment‑variable overrides.
- **Tests** – Basic unit tests for ingestion, retrieval, and response generation.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python      | >=3.9   |
| pip         | latest  |
| OpenAI API key (or compatible provider) | – |
| Git (optional) | – |

> **Note**: The project is LLM‑agnostic. Replace `OpenAI` calls with `AzureOpenAI`, `Anthropic`, `Cohere`, etc., by adjusting the `LLMProvider` class.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/FrancoisBib/langchain-chatbot.git
cd langchain-chatbot

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # on Windows use `.venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

If you prefer Poetry:

```bash
poetry install
```

---

## Configuration

All configurable values are stored in `config.py` and can be overridden with environment variables. The most common variables are:

```dotenv
# .env file (place in project root)
OPENAI_API_KEY=sk-xxxxxxxxxxxx
EMBEDDING_MODEL=text-embedding-ada-002
LLM_MODEL=gpt-4o-mini
VECTOR_STORE=faiss   # options: faiss, chroma, pinecone
DOCS_PATH=./data    # folder containing source documents
```

Load the `.env` file automatically with `python-dotenv` (already a dependency).

---

## Running the Chatbot

### CLI mode

```bash
python -m chatbot.cli
```

You will be prompted for a question. The bot will retrieve the most relevant chunks, feed them to the LLM, and print the answer.

### FastAPI server

```bash
uvicorn chatbot.api:app --reload
```

The API exposes a single endpoint:

- `POST /chat` – body `{ "question": "Your query" }`
- Returns `{ "answer": "Generated response", "sources": ["doc1.txt:12-34", ...] }`

You can test it with `curl` or any HTTP client:

```bash
curl -X POST http://127.0.0.1:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"question": "Explain RAG in simple terms"}'
```

---

## Project Structure

```
langchain-chatbot/
├─ chatbot/                # Core package
│   ├─ __init__.py
│   ├─ ingestion.py        # Document loaders & chunking
│   ├─ vector_store.py     # FAISS/Chroma abstraction
│   ├─ rag.py              # Retrieval + generation pipeline
│   ├─ cli.py              # Command‑line interface
│   └─ api.py              # FastAPI entry point
├─ tests/                  # Unit tests
│   ├─ test_ingestion.py
│   ├─ test_vector_store.py
│   └─ test_rag.py
├─ data/                   # Example documents (git‑ignored)
├─ .env.example            # Template for environment variables
├─ requirements.txt
├─ pyproject.toml          # Poetry metadata (optional)
└─ README.md               # <‑‑ you are here
```

---

## How RAG Works in This Project

1. **Ingestion** – Files from `DOCS_PATH` are loaded, cleaned, and split into overlapping chunks (default: 1000 characters, 200‑char overlap).
2. **Embedding** – Each chunk is embedded using the configured embedding model. The embeddings are stored in the chosen vector store.
3. **Retrieval** – At query time, the user question is embedded, and the top‑k most similar chunks are fetched.
4. **Prompt Construction** – A system prompt describes the chatbot’s role. Retrieved chunks are inserted into a *context* variable that the LLM sees.
5. **Generation** – The LLM produces a response that is grounded in the retrieved documents, reducing hallucinations.
6. **Source Attribution** – The API returns the source file names and page/line numbers for each chunk, enabling transparency.

---

## Testing

Run the test suite with:

```bash
pytest -q
```

The tests cover:
- Document loading and chunking logic
- Vector store indexing and similarity search
- End‑to‑end RAG pipeline (mocked LLM for speed)

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository.
2. **Create a branch** for your feature or bug‑fix:
   ```bash
   git checkout -b feature/awesome‑feature
   ```
3. **Write tests** for any new functionality.
4. **Run linting & formatting**:
   ```bash
   pip install pre-commit
   pre-commit run --all-files
   ```
5. **Submit a Pull Request** with a clear description of the change.

### Code Style
- Use **black** for formatting.
- Follow **PEP 8** and **flake8** warnings.
- Type‑hint all public functions (mypy compliance).

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- The **LangChain** team for the powerful abstractions.
- OpenAI for the embedding and LLM APIs.
- The open‑source community for vector‑store implementations.
