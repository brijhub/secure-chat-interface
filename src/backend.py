"""FastAPI backend. Run: python -m uvicorn backend:app --app-dir src"""

import logging
from copy import deepcopy
from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from config import MAX_HISTORY_MESSAGES, USER_ACCESS
from history import init_history, load_history, append_messages, clear_history
from rag import answer_question, load_retriever
from inference import start_inference, stop_inference
from ingest import ensure_index


# Keep turns ordered within a conversation without blocking other users.
conversation_locks = {}
lock_registry_guard = Lock()


def get_conversation_lock(key):
    with lock_registry_guard:
        return conversation_locks.setdefault(key, Lock())


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    init_history()
    logger.info("Preparing PDF index and loading local models. First startup may take a few minutes.")
    try:
        start_inference()
        ensure_index()
        load_retriever()
        logger.info("Index and models ready.")
        yield
    finally:
        stop_inference()


app = FastAPI(title="Secure Chat Interface", lifespan=lifespan)


class SessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    email: str = Field(min_length=1, max_length=254)
    session_id: str = Field(min_length=1, max_length=128)


class ChatRequest(SessionRequest):
    question: str = Field(min_length=1, max_length=4000)


def conversation_key(email: str, session_id: str) -> tuple[str, str]:
    email = email.strip().lower()
    if not USER_ACCESS.get(email):
        raise PermissionError("Unknown user or no company access.")
    if not session_id or not session_id.strip():
        raise ValueError("session_id cannot be empty.")
    return email, session_id


def chat(email: str, session_id: str, question: str) -> dict:
    """Keep the full transcript, using only eligible recent messages as context."""
    key = conversation_key(email, session_id)
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    with get_conversation_lock(key):
        history = load_history(*key)
        context = deepcopy([
            message for message in history if message.get("include_in_context", True)
        ][-MAX_HISTORY_MESSAGES:])
        try:
            result = answer_question(question, key[0], context)
        except Exception as error:
            detail = (str(error) if isinstance(error, (PermissionError, ValueError))
                      else "Could not process the question. Check backend logs.")
            append_messages(*key, [
                {"role": "user", "content": question, "include_in_context": False},
                {"role": "assistant", "content": detail, "sources": [],
                 "include_in_context": False},
            ])
            raise
        include_in_context = result.get("include_in_context", True)
        append_messages(*key, [
            {"role": "user", "content": question,
             "include_in_context": include_in_context},
            {"role": "assistant", "content": result["answer"],
             "sources": deepcopy(result["sources"]),
             "include_in_context": include_in_context},
        ])
    return result


def get_history(email: str, session_id: str) -> list[dict]:
    """Load the full transcript for this conversation."""
    key = conversation_key(email, session_id)
    with get_conversation_lock(key):
        return load_history(*key)


def reset_conversation(email: str, session_id: str) -> None:
    """Clear only this user/session; repeated resets are harmless."""
    key = conversation_key(email, session_id)
    with get_conversation_lock(key):
        clear_history(*key)


@app.get("/health")
def health():
    """Served after startup has prepared the index and loaded both models."""
    return {"status": "ok"}


@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    # Sync routes run outside the event loop; CPU generation can block this worker.
    try:
        return chat(request.email, request.session_id, request.question)
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception("Chat processing failed")
        raise HTTPException(status_code=500, detail="Could not process the question. Check backend logs.") from error


@app.post("/history")
def history_endpoint(request: SessionRequest):
    try:
        return {"messages": get_history(request.email, request.session_id)}
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/reset")
def reset_endpoint(request: SessionRequest):
    try:
        reset_conversation(request.email, request.session_id)
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"status": "reset"}
