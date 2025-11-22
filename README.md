# LangChain Chatbot

**LangChain Chatbot** is a modular, extensible framework for building AI‑powered conversational agents using the LangChain ecosystem. It demonstrates how to combine large language models (LLMs) with Retrieval‑Augmented Generation (RAG) techniques to create knowledgeable, context‑aware chatbots.

---

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Demo](#running-the-demo)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Extending the Bot](#extending-the-bot)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **LangChain Integration** – Leverages LangChain chains, agents, and memory.
- **Retrieval‑Augmented Generation (RAG)** – Uses vector stores (e.g., Chroma, FAISS) to retrieve relevant documents and feed them to the LLM.
- **Modular Design** – Separate modules for data ingestion, indexing, retrieval, and chat handling.
- **Configurable LLMs** – Works with OpenAI, Anthropic, Cohere, HuggingFace, etc., via LangChain's LLM wrappers.
- **Extensible Prompt Templates** – Easy to customize system and user prompts.
- **Docker Support** – Containerised development and deployment.

---

## Getting Started

### Prerequisites

- Python **3.9** or newer
- An API key for the LLM provider you plan to use (e.g., `OPENAI_API_KEY`)
- Optional: Docker & Docker‑Compose if you prefer containerised execution

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/langchain-chatbot.git
cd langchain-chatbot

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

If you want to run the project inside Docker:

```bash
docker compose up --build
```

### Running the Demo

The repository ships with a minimal demo that loads a small document collection, builds a vector store, and starts a CLI chat interface.

```bash
python -m chatbot.main
```

You will be prompted for a query; the bot will retrieve relevant passages, augment the LLM prompt, and return a response.

---

## Project Structure

```
langchain-chatbot/
├─ chatbot/                     # Core package
│  ├─ __init__.py
│  ├─ ingestion.py              # Document loading & chunking
│  ├─ indexing.py               # Vector store creation (FAISS/Chroma)
│  ├─ retrieval.py              # Retrieval logic (similarity search)
│  ├─ prompts.py                # Prompt templates & utilities
│  ├─ chain.py                  # LangChain chain that ties LLM + retrieval
│  └─ main.py                   # CLI entry point / demo script
├─ tests/                       # Unit & integration tests
├─ docs/                        # Additional documentation (optional)
├─ .env.example                 # Example environment variables file
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
└─ README.md
```

---

## Configuration

Configuration is driven by environment variables. Copy `.env.example` to `.env` and fill in the required values:

```dotenv
# LLM provider configuration (choose one)
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
COHERE_API_KEY=your-cohere-key

# Vector store selection (FAISS or CHROMA)
VECTOR_STORE=FAISS   # or CHROMA

# Optional: Path to your document collection
DATA_PATH=./data
```

The `chatbot.config` module reads these variables and creates the appropriate LangChain objects.

---

## Usage Examples

### 1. Simple CLI Chat

```bash
python -m chatbot.main
```

### 2. Using the Bot as a Library

```python
from chatbot.chain import get_chat_chain
from langchain.schema import HumanMessage

# Initialise the chain (automatically loads the vector store)
chat_chain = get_chat_chain()

# Send a message and get a response
response = chat_chain.invoke([HumanMessage(content="Explain the concept of RAG.")])
print(response.content)
```

### 3. Custom Prompt Template

```python
from chatbot.prompts import SYSTEM_PROMPT, USER_PROMPT

CUSTOM_SYSTEM = "You are a helpful assistant specialized in finance."
CUSTOM_USER = "{question}\nRelevant context:\n{retrieved_docs}"

# Pass custom templates when building the chain
chat_chain = get_chat_chain(
    system_prompt=CUSTOM_SYSTEM,
    user_prompt=CUSTOM_USER,
)
```

---

## Extending the Bot

- **Add New Document Sources** – Implement a loader in `ingestion.py` (e.g., PDFs, webpages, Notion). Use LangChain's `DocumentLoader` subclasses.
- **Swap Vector Stores** – The `indexing.py` module abstracts the store; add support for Pinecone, Weaviate, or Milvus by extending `VectorStoreFactory`.
- **Advanced Memory** – Replace the simple stateless chain with a `ConversationBufferMemory` or a custom memory class to retain chat history.
- **Deploy as API** – Wrap `get_chat_chain` in a FastAPI endpoint for production use.

---

## Testing

The project includes a test suite based on `pytest`. Run the tests with:

```bash
pytest -v
```

All tests are located in the `tests/` directory and cover ingestion, indexing, retrieval, and chain execution.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/awesome-feature`).
3. Ensure code follows the existing style (PEP 8, type hints).
4. Add or update tests for new functionality.
5. Run the full test suite (`pytest`).
6. Submit a pull request with a clear description of the changes.

For major changes, open an issue first to discuss the proposed design.

---

## License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.
