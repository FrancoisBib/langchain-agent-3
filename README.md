# LangChain Chatbot

## 📖 Overview

**LangChain‑Chatbot** is a minimal yet production‑ready reference implementation of a Retrieval‑Augmented Generation (RAG) chatbot built on top of **[LangChain](https://python.langchain.com/)**.  It demonstrates how to combine:

- **Large Language Models (LLMs)** for natural‑language generation
- **Vector stores** for semantic retrieval of documents
- **Chains & agents** to orchestrate the retrieval‑generation workflow
- **FastAPI** (or optionally Streamlit) as a thin HTTP layer for integration with UI front‑ends or other services

The repository is deliberately kept simple so that developers can:

1. **Learn** the core concepts of LangChain and RAG.
2. **Bootstrap** their own chatbot projects with a solid foundation.
3. **Contribute** improvements, new integrations, or tests.

---

## 🛠️ Features

- 📚 **Document ingestion** – Supports PDF, TXT, Markdown, and CSV files.
- 🔍 **Semantic search** – Uses OpenAI embeddings (or any HuggingFace embedding model) stored in a ChromaDB vector store.
- 🤖 **RAG pipeline** – Retrieves the most relevant chunks and passes them to an LLM (OpenAI `gpt‑4o`, Anthropic `claude‑3`, etc.) with a concise system prompt.
- ⚡ **Streaming responses** – Optional token‑wise streaming for a more interactive UI experience.
- 🧩 **Modular architecture** – All components (loader, splitter, embedder, vector store, LLM) are interchangeable via LangChain interfaces.
- 🧪 **Testing utilities** – Pytest fixtures for unit‑testing the retrieval and generation steps.
- 📦 **Docker support** – Multi‑stage Dockerfile for reproducible local development and production deployment.

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python      | `>=3.9` |
| Docker      | `>=20.10` |
| OpenAI API key (or alternative LLM provider) |

### 1. Clone the repository

```bash
git clone https://github.com/your‑org/langchain-chatbot.git
cd langchain-chatbot
```

### 2. Install dependencies (virtual‑env recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set environment variables

Create a `.env` file at the project root (or export variables in your shell):

```dotenv
# LLM provider – currently supported: openai, anthropic, together
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-*****
# Optional – change the model
OPENAI_MODEL=gpt-4o-mini
# Vector store configuration (defaults to local ChromaDB)
CHROMA_PERSIST_DIR=./chroma_db
```

### 4. Ingest your knowledge base

Place any documents you want the bot to know about inside the `data/` directory, then run:

```bash
python scripts/ingest.py
```

The script will:
1. Load files using LangChain loaders.
2. Split them into chunks (default: 1 000 tokens, 200 token overlap).
3. Compute embeddings and persist them in the Chroma vector store.

### 5. Run the API server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The chatbot endpoint is now available at `POST /chat`:

```json
# Request payload
{
  "message": "Explain the difference between supervised and unsupervised learning."
}
```

The response contains the generated answer and the source documents used for retrieval.

---

## 📚 Detailed Usage

### 5.1 API contract

| Method | Endpoint | Body | Returns |
|--------|----------|------|---------|
| `POST` | `/chat` | `{ "message": "string" }` | `{ "answer": "string", "sources": [{"page_content": "...", "metadata": {...}}] }` |
| `GET`  | `/health` | – | `{ "status": "ok" }` |

### 5.2 Customising the Retrieval Chain

The core RAG logic lives in `app/rag.py`.  To swap components, modify the factory functions:

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# 1️⃣ Embeddings – replace with HuggingFaceEmbeddings if you prefer an open‑source model
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# 2️⃣ Vector store – any LangChain‑compatible store works (FAISS, Pinecone, Weaviate, …)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 3️⃣ LLM – change model or provider
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# 4️⃣ RetrievalQA chain – you can tweak the prompt template here
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)
```

### 5.3 Streaming responses (optional)

If you enable streaming in the `.env` (`STREAMING=true`), the FastAPI endpoint will use Server‑Sent Events (SSE).  Front‑ends can consume the stream token‑by‑token for a chat‑like experience.

---

## 🧪 Testing

```bash
pytest -q
```

The test suite covers:
- Document loading & splitting
- Vector store persistence & similarity search
- End‑to‑end RAG generation (mocked LLM calls)

Add new tests under `tests/` to protect future contributions.

---

## 🐳 Docker

### Build the image

```bash
docker build -t langchain-chatbot:latest .
```

### Run the container (environment variables can be passed with `-e`)

```bash
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -e OPENAI_API_KEY=sk-***** \
  langchain-chatbot:latest
```

The container starts the FastAPI server automatically.

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository.
2. **Create a branch** for your feature or bug‑fix (`git checkout -b feat/awesome‑feature`).
3. **Write tests** for any new functionality.
4. **Run the full test suite** (`pytest`).
5. **Submit a Pull Request** with a clear description of the change.

### Code style
- Use **Black** for formatting (`black .`).
- Type hints are mandatory (`mypy` passes).
- Lint with **ruff** (`ruff check .`).

---

## 📜 License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.

---

## 📚 Further Reading

- LangChain Documentation – <https://python.langchain.com/docs/>
- Retrieval‑Augmented Generation – <https://arxiv.org/abs/2005.11401>
- OpenAI Embeddings – <https://platform.openai.com/docs/guides/embeddings>

---

*Happy coding!*