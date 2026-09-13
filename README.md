# Secure Chat Interface

Chat with company PDFs using simulated user access, cited answers, and saved conversations.

## 1. Setup

Clone this repository, Use Python 3.13:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Ingest PDFs
First-time setup needs internet to download models.
Build the local vector database from the supplied PDFs in `data/`:

```powershell
.\.venv\Scripts\python.exe src/ingest.py
```

Database files are stored locally in `storage/` and excluded from Git. Ingestion creates `storage/chroma_db/`; the backend creates `storage/chat_history.sqlite3` on startup.

## 3. Start backend

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend:app --app-dir src
```

Run this command from the project folder; `--app-dir src` makes the source modules available for imports. Wait for **Application startup complete**.

## 4. Start UI

In a second PowerShell terminal, from the project folder:

```powershell
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe -m streamlit run src/app.py
```

Open http://localhost:8501 and select a user:

| User | Allowed companies |
| --- | --- |
| alice@email.com | Meta |
| bob@email.com | Meta, Google |
| charlie@email.com | TCS, Amazon |

Try: **What was Meta's revenue in Q1 2025?** Then: **How did it compare with last year?**

Chats survive logout and restarts; **Reset Conversation** deletes the current chat. Permissions and settings are in `src/config.py`.

## Validation

```powershell
.\.venv\Scripts\python.exe validation/evaluate.py
```

Uses `validation/validation_data.json` and writes `validation/results.json`: retrieval success, timings, and access checks—not LLM answer accuracy.

Validation covers 80 cases across the four supplied company PDFs.

### Latest results (13 September 2026)

On the local development dataset, **64/68 questions (94.12%)** retrieved a labelled relevant PDF page within the top five chunks; the first chunk matched a relevant page for **47/68 (69.12%)**. All **4 explicit access-denial checks passed**.

| Metric | k = 3 (test only) | k = 5 (application default) |
| --- | --- | --- |
| Hit Rate@k | 91.18% (62/68) | 94.12% (64/68) |
| Precision@k | 42.16% | 30.00% |
| Recall@k | 91.18% | 94.12% |
| MRR@k | 0.7892 | 0.7966 |

Relevance uses **page labels**, not verified chunk evidence. Precision measures relevant chunk slots divided by k; recall measures distinct labelled pages found; MRR measures the reciprocal rank of the first relevant chunk (zero for a miss).

**Misses:** Four questions missed their labelled pages: Amazon Q1 2025 operating income and net income, Alphabet's Q1 2025 unrealized equity gain, and Meta's Q2 2025 revenue forecast. These are retrieval misses against the dataset labels, not verified wrong answers.

**Scope:** This is a retrieval score, not overall RAG answer accuracy. Follow-ups use prepared standalone questions; generated answers, citation support, and the eight unanswerable cases are not scored. Full results: [validation/results.json](validation/results.json).

## Components

- **Embedding model (`all-MiniLM-L6-v2`):** Converts PDF text and questions into vectors to find passages with similar meaning.
- **Language model (`Qwen2.5-0.5B-Instruct`):** Generates answers from retrieved passages locally on CPU, without an API key.
- **ChromaDB:** Stores document vectors and metadata to retrieve relevant passages filtered by user access.
- **SQLite:** Stores chat history locally so conversations survive restarts.
- **FastAPI:** Provides backend API endpoints for chat, user access checks, and conversation history.
- **Streamlit:** Provides the browser interface for selecting a user, asking questions, and viewing answers.

## Limitations and further improvements

- **Model serving:** This demo uses a local thread pool. A dedicated serving engine such as vLLM or Ollama could be evaluated for concurrent requests, depending on the hardware and model but purposely I dropped it for this.
- **Unsupported questions:** The current prompt uses hardcoded and dummy methods to decline unsupported questions, but can implement some good fiktering techniques based on question intents.
- **Conversation memory:** FOr this I'm using only the last six eligible messages from history for response generation. Token-based history limits, conversation summaries, and other techniqies can be used to support longer conversations.
