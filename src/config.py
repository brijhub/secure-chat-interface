"""Shared settings and simulated user permissions."""

from pathlib import Path


# Resolve paths from this file so they do not depend on the launch directory.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"
STORAGE_PATH = BASE_DIR / "storage"
CHROMA_PATH = STORAGE_PATH / "chroma_db"
HISTORY_DB_PATH = STORAGE_PATH / "chat_history.sqlite3"

# Company identifiers match the PDF filenames without the .pdf extension.
USER_ACCESS = {
    "alice@email.com": ["meta"],
    "bob@email.com": ["meta", "google"],
    "charlie@email.com": ["tcs", "wipro","amazon"],
}

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COMPANY_ALIASES = {
    "amazon": ["amazon", "aws"],
    "google": ["google", "alphabet"],
    "meta": ["meta", "facebook"],
    "tcs": ["tcs", "tata consultancy services"],
    "wipro": ["wipro"],
}
COMPANY_FUZZY_THRESHOLD = 0.85  # Spelling similarity, not semantic confidence.

# Small instruction-tuned model for local CPU inference.
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LLM_MAX_INPUT_TOKENS = 4096
LLM_MAX_NEW_TOKENS = 256
LLM_PARALLEL_REQUESTS = 3
LLM_CPU_THREADS = 2  # Per-operation CPU threads; leave capacity for UI/retrieval.

COLLECTION_NAME = "earnings_documents"
TOP_K = 5
CHUNK_SIZE = 220  # MiniLM word-piece tokens, excluding special tokens.
CHUNK_OVERLAP = 40
EMBEDDING_BATCH_SIZE = 32
MAX_HISTORY_MESSAGES = 6  # Eligible model-context messages; display history is not trimmed.
BACKEND_URL = "http://127.0.0.1:8000"
API_TIMEOUT_SECONDS = 300
