# LangChain Chatbot with Retrieval-Augmented Generation (RAG)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

A minimal yet extensible example of building a **chatbot** powered by **LangChain** and **Retrieval‑Augmented Generation (RAG)**.  The project demonstrates how to:

- Load documents from a variety of sources (PDF, TXT, CSV, web pages, etc.)
- Create a vector store (FAISS, Chroma, Pinecone, …) for semantic search
- Combine a language model (OpenAI, Anthropic, Llama‑2, etc.) with retrieved context
- Deploy the bot locally or as an API endpoint (FastAPI) and optionally expose a UI (Streamlit/Gradio)

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Bot](#running-the-bot)
- [Project Structure](#project-structure)
- [Extending & Contributing](#extending--contributing)
- [Testing](#testing)
- [License](#license)

---

## Features

- **RAG pipeline**: retrieve relevant chunks with similarity search, then feed them to a LLM for generation.
- **Modular components**: interchangeable document loaders, vectorstores, and LLM wrappers.
- **Prompt engineering**: ready‑made system prompts and a `ChatPromptTemplate` that can be customised.
- **Streaming responses**: optional token‑by‑token streaming for a more interactive UI.
- **Docker support**: `Dockerfile` and `docker-compose.yml` for reproducible environments.
- **Extensive type hints & docstrings** – IDE‑friendly.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/langchain-chatbot.git
cd langchain-chatbot

# Install dependencies in a virtual environment
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt

# Set your environment variables (see Configuration below)
cp .env.example .env
# edit .env with your API keys

# Index your documents (one‑time operation)
python scripts/create_index.py data/

# Run the chatbot (FastAPI server)
uvicorn app.main:app --reload
```

Open your browser at `http://127.0.0.1:8000/docs` to explore the OpenAPI UI or integrate the endpoint into your front‑end.

---

## Installation

### Prerequisites
- Python **3.9+**
- Access to an LLM provider (OpenAI, Anthropic, Cohere, etc.) – API key required.
- (Optional) GPU for faster embedding generation – otherwise CPU works fine.

### Dependencies
All required packages are listed in `requirements.txt`.  The core stack includes:

- `langchain`
- `langchain-community`
- `faiss-cpu` (or `faiss-gpu` if you have CUDA)
- `pydantic`
- `fastapi` & `uvicorn`
- `python-dotenv`
- `tiktoken` (for token counting)

Run the installation command shown in the Quick Start section.

---

## Configuration

The project uses a **.env** file (managed by `python-dotenv`).  Copy the example file and fill in your values:

```dotenv
# .env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
# Choose an embedding model – e.g., text-embedding-ada-002
EMBEDDING_MODEL=text-embedding-ada-002
# Vectorstore – faiss (default), chroma, pinecone, weaviate, etc.
VECTORSTORE=faiss
# Optional: Pinecone credentials if you use Pinecone
PINECONE_API_KEY=xxxx
PINECONE_ENV=us-east1-gcp
# FastAPI settings
HOST=0.0.0.0
PORT=8000
```

### Switching LLMs
The `LLMFactory` in `app/llm.py` reads `LLM_PROVIDER` and `LLM_MODEL` from the environment.  Supported providers are:
- `openai`
- `anthropic`
- `cohere`
- `huggingface`

Add the corresponding API keys to `.env`.

---

## Running the Bot

### Development server (FastAPI)
```bash
uvicorn app.main:app --reload
```
Visit `http://127.0.0.1:8000/docs` for the Swagger UI.

### Streamlit UI (optional)
```bash
streamlit run ui/chat_interface.py
```
The UI demonstrates streaming responses and shows the retrieved document snippets.

### Docker
```bash
docker compose up --build
```
The service will be reachable at `http://localhost:8000`.

---

## Project Structure

```
langchain-chatbot/
├─ app/                     # FastAPI application
│   ├─ main.py              # entry point, router registration
│   ├─ routes.py            # /chat endpoint implementation
│   ├─ rag.py               # RAG pipeline (retriever + generator)
│   └─ llm.py               # LLM factory & wrappers
├─ data/                    # Sample documents (place your own here)
├─ scripts/                 # Utility scripts (index creation, cleanup)
│   └─ create_index.py
├─ ui/                      # Optional front‑ends (Streamlit / Gradio)
│   └─ chat_interface.py
├─ tests/                   # Unit & integration tests
│   └─ test_rag.py
├─ .env.example             # Template for environment variables
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
└─ README.md                # ← this file
```

---

## Extending & Contributing

### Adding a new document loader
1. Create a class inheriting from `langchain.document_loaders.base.BaseLoader`.
2. Implement `load()` returning a list of `Document` objects.
3. Register the loader in `scripts/create_index.py` under the `SUPPORTED_LOADERS` dictionary.

### Supporting a new vectorstore
- Install the appropriate Python client (e.g., `pip install chromadb`).
- Extend `app/vectorstore.py` with a factory method that returns an instance implementing `langchain.vectorstores.VectorStore`.

### Contributing guide
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Write tests for your changes (`pytest`).
4. Ensure linting passes (`ruff check .`).
5. Open a Pull Request with a clear description.

Please adhere to the **PEP 8** style guide and include docstrings for any public function or class.

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest -vv
```
The test suite covers:
- Document loading & chunking
- Vectorstore indexing & similarity search
- End‑to‑end RAG generation (mocked LLM)

Continuous Integration runs on GitHub Actions on every PR.

---

## License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.

---

## Acknowledgements

- The **LangChain** community for the powerful abstractions.
- OpenAI, Anthropic, and other LLM providers for accessible APIs.
- The authors of the underlying vector databases.
