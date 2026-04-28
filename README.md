# rag-doc-chat

A RAG-powered document chat assistant built with LangChain and Claude.
Upload a document and ask questions about it in plain English. Answers are grounded in your document, not Claude's general knowledge.

## Demo

1. Start the API server
2. Open the Streamlit frontend
3. Upload a PDF, Word doc, or text file
4. Ask questions in the chat interface

## Features

- Upload `.txt`, `.pdf`, `.docx`, and `.md` files
- Google Docs support via URL
- Streaming responses that appear word by word
- Conversation memory per session
- Strict hallucination prevention — Claude only answers from your document
- Health check and document upload REST endpoints
- Eval suite for regression testing

## Tech stack

- [LangChain](https://python.langchain.com/) — RAG pipeline and chain orchestration
- [Claude Haiku](https://anthropic.com) — language model
- [Voyage AI](https://www.voyageai.com/) — text embeddings
- [FAISS](https://github.com/facebookresearch/faiss) — local vector store
- [FastAPI](https://fastapi.tiangolo.com/) — REST API backend
- [Streamlit](https://streamlit.io/) — chat frontend
- Python 3.10+

## Project structure

```
rag-doc-chat/
├── final/                  # Complete application
│   ├── main.py             # FastAPI backend
│   ├── app.py              # Streamlit frontend
│   ├── docs/               # Upload documents here
│   ├── .env.example        # Required API keys
│   └── .gitignore
├── module_01/              # LangChain & LLM fundamentals
├── module_02/              # Prompt templates & output parsers
├── module_03/              # Chains & LCEL
├── module_04/              # Memory & stateful conversations
├── module_05/              # Embeddings, vector stores & RAG
├── module_06/              # Serving LangChain apps as APIs
├── module_07/              # Agents & tools
└── module_08/              # Evaluation, debugging & production quality
```

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/hntrjoseph28-rgb/rag-doc-chat.git
cd rag-doc-chat/final
```

### 2. Install dependencies

```bash
pip install langchain langchain-anthropic langchain-community langchain-voyageai langchain-text-splitters faiss-cpu fastapi uvicorn streamlit pypdf docx2txt python-dotenv
```

### 3. Set up API keys

Copy `.env.example` to `.env` and fill in your keys: 
ANTHROPIC_API_KEY=your_anthropic_key_here
VOYAGE_API_KEY=your_voyage_key_here

- Get an Anthropic API key at [console.anthropic.com](https://console.anthropic.com)
- Get a Voyage AI API key at [dashboard.voyageai.com](https://dashboard.voyageai.com)

### 4. Run the API server

```bash
cd final
python -m uvicorn main:app --reload
```

### 5. Run the Streamlit frontend

In a second terminal:

```bash
cd final
python -m streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check server status |
| POST | `/upload` | Upload a document |
| POST | `/load-gdoc` | Load a Google Doc by URL |
| POST | `/ask/stream` | Ask a question (streaming) |

## How it works

1. A document is uploaded and split into chunks using `RecursiveCharacterTextSplitter`
2. Each chunk is embedded using Voyage AI and stored in a FAISS vector store
3. When a question is asked, the most relevant chunks are retrieved by similarity search
4. The chunks are passed to Claude as context along with the conversation history
5. Claude answers using only the provided context — if the answer isn't there, it says so

## Status

Complete: Built as a structured 8-module learning project covering LangChain fundamentals through production deployment.
