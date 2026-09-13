# Secure Chat Interface

Chat with company PDFs using simulated user access, cited answers, and saved conversations. Runs locally on CPU; no API key required.

## 1. Setup

Clone this repository, then open PowerShell in the project folder. Use Python 3.13:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

First-time setup needs internet to download models.

## 2. Ingest PDFs

Build the local vector database from the supplied PDFs in `data/`:

```powershell
.\.venv\Scripts\python.exe src/ingest.py
```

Database files are stored locally in `storage/` and excluded from Git. Ingestion creates `storage/chroma_db/`; the backend creates `storage/chat_history.sqlite3` on startup.

## 3. Start backend

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.backend:app
```

Wait for **Application startup complete**.

## 4. Start UI

In a second PowerShell terminal, from the project folder:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/app.py
```

Open http://localhost:8501 and select a user:

| User | Allowed companies |
| --- | --- |
| alice@email.com | Meta |
| bob@email.com | Meta, Google |
| charlie@email.com | TCS, Amazon |

Try: **What was Meta's revenue in Q1 2025?** Then: **How did it compare with last year?**

Use separate tabs for simultaneous users. Chats survive logout and restarts; **Reset Conversation** deletes the current chat. Permissions and settings are in `src/config.py`.

## Validation

```powershell
.\.venv\Scripts\python.exe validation/evaluate.py
```

Uses `validation/validation_data.json` and writes `validation/results.json`: retrieval success, timings, and access checks—not LLM answer accuracy.

Validation covers 80 cases across the four supplied company PDFs.
