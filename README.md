# LangChain Chatbot

## 📖 Overview

**LangChain‑Chatbot** is a lightweight, extensible reference implementation of a Retrieval‑Augmented Generation (RAG) chatbot built on top of **[LangChain](https://github.com/langchain-ai/langchain)**. It demonstrates how to combine:

- **Large Language Models (LLMs)** for natural‑language generation
- **Vector stores** for semantic document retrieval
- **Chains & agents** for orchestrating complex workflows

The project is deliberately minimal so you can focus on the core concepts, experiment with different components, and use it as a starting point for your own production‑grade chatbot.

---

## ✨ Features

- ✅ **Modular architecture** – interchangeable LLMs, embeddings, and vector stores.
- ✅ **RAG pipeline** – retrieve relevant chunks, augment the prompt, and generate a response.
- ✅ **Streaming support** – optional token‑by‑token streaming for UI integration.
- ✅ **Docker‑ready** – containerised development and deployment.
- ✅ **Extensive type‑hints & docstrings** – IDE‑friendly and easy to extend.
- ✅ **Test suite** – unit tests for the core retrieval and generation logic.

---

## 🚀 Quick Start

### Prerequisites

- Python **3.10+**
- An OpenAI API key (or any other LLM provider supported by LangChain)
- Optional: `docker` & `docker‑compose` if you prefer containerised execution

### 1. Clone the repository

```bash
git clone https://github.com/your‑org/langchain-chatbot.git
cd langchain-chatbot
```

### 2. Install dependencies

We recommend using a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set environment variables

Create a `.env` file in the project root:

```dotenv
# .env
OPENAI_API_KEY=sk-****************
# Choose the vector store you want (default: chroma)
VECTOR_STORE=chroma
```

> **Tip:** The project also works with `FAISS`, `Pinecone`, `Weaviate`, etc. See the *Configuration* section below.

### 4. Run the demo script

```bash
python -m chatbot.run
```

You will be prompted for a question; the bot will retrieve relevant documents from the default `data/` folder, augment the prompt, and stream the answer.

---

## 📦 Installation as a Library

If you want to embed the chatbot in your own application, install it via pip:

```bash
pip install git+https://github.com/your‑org/langchain-chatbot.git
```

Then import the high‑level helper:

```python
from langchain_chatbot import Chatbot

bot = Chatbot()
answer = bot.ask("Explain the difference between supervised and unsupervised learning.")
print(answer)
```

---

## 🛠️ Architecture & Core Concepts

### 1. **Embedding & Vector Store**

- **Embeddings** – `OpenAIEmbeddings` is the default, but any `Embedding` implementation from LangChain can be swapped (e.g., `HuggingFaceEmbeddings`).
- **Vector Store** – `Chroma` is used for local development. The `VectorStore` abstraction lets you switch to a remote service with a single config change.

### 2. **Retriever**

The `Retriever` wraps the vector store and returns the top‑`k` most relevant document chunks based on cosine similarity.

### 3. **Prompt Template**

A `ChatPromptTemplate` injects the retrieved context into a system prompt that instructs the LLM to answer concisely and cite sources when possible.

### 4. **LLM Chain**

The `LLMChain` combines the prompt template with the chosen LLM (`OpenAI`, `AzureOpenAI`, `ChatAnthropic`, …). Streaming is enabled by passing `streaming=True`.

### 5. **Chatbot Facade**

`Chatbot` (exposed in `langchain_chatbot/__init__.py`) hides the plumbing:

```python
class Chatbot:
    def __init__(self, *, llm=None, retriever=None, prompt=None, top_k=4):
        ...
    def ask(self, query: str) -> str:
        ...
```

---

## ⚙️ Configuration

All settings can be overridden via environment variables or explicit arguments when constructing `Chatbot`.

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_MODEL` | Name of the OpenAI model (e.g., `gpt-4o-mini`) | `gpt-3.5-turbo` |
| `EMBEDDING_MODEL` | Embedding model identifier | `text-embedding-3-large` |
| `VECTOR_STORE` | `chroma`, `faiss`, `pinecone`, `weaviate` | `chroma` |
| `TOP_K` | Number of retrieved chunks per query | `4` |
| `DATA_PATH` | Directory containing source markdown/pdf files | `data/` |

---

## 🧪 Testing

```bash
pytest -q
```

The test suite covers:
- Retrieval correctness (vector similarity)
- Prompt rendering
- End‑to‑end generation with a mock LLM

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository.
2. Create a **feature branch** (`git checkout -b feat/awesome‑feature`).
3. Write **unit tests** for new functionality.
4. Ensure the test suite passes (`pytest`).
5. Submit a **Pull Request** with a clear description of the change.

### Code Style

- Use **black** for formatting (`black .`).
- Type hints are required for all public functions.
- Follow the existing folder layout:
  - `chatbot/` – core library
  - `scripts/` – CLI utilities
  - `tests/` – test suite

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 📚 Further Reading

- LangChain Documentation: https://python.langchain.com/
- Retrieval‑Augmented Generation Primer: https://arxiv.org/abs/2005.11401
- OpenAI API Reference: https://platform.openai.com/docs/api-reference/introduction
