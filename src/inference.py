"""Small in-process inference pool sharing one read-only CPU model."""

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import logging
import time
from uuid import uuid4

if __package__:
    from .config import LLM_MAX_NEW_TOKENS, LLM_PARALLEL_REQUESTS, LLM_CPU_THREADS
    from .model import load_model, generate_text as generate_local
else:
    from config import LLM_MAX_NEW_TOKENS, LLM_PARALLEL_REQUESTS, LLM_CPU_THREADS
    from model import load_model, generate_text as generate_local

_pool = None
_pool_lock = Lock()
logger = logging.getLogger("uvicorn.error")


def start_inference():
    global _pool
    with _pool_lock:
        if _pool is None:
            import torch

            if LLM_PARALLEL_REQUESTS < 1 or LLM_CPU_THREADS < 1:
                raise ValueError("Inference workers and CPU threads must be positive.")
            torch.set_num_threads(LLM_CPU_THREADS)
            load_model()
            # Initialize lazy generation state before accepting parallel calls.
            generate_local([{"role": "user", "content": "Hello"}], 1)
            _pool = ThreadPoolExecutor(
                max_workers=LLM_PARALLEL_REQUESTS, thread_name_prefix="llm",
            )


def stop_inference():
    global _pool
    with _pool_lock:
        pool, _pool = _pool, None
    if pool is not None:
        pool.shutdown(wait=True)


def _generate(messages, max_new_tokens):
    request_id = uuid4().hex[:8]
    started = time.perf_counter()
    logger.info("Inference %s START", request_id)
    try:
        return generate_local(messages, max_new_tokens)
    finally:
        logger.info("Inference %s END %.2fs", request_id, time.perf_counter() - started)


def generate_text(messages, max_new_tokens=LLM_MAX_NEW_TOKENS):
    # Lazy startup also supports the standalone RAG/evaluation scripts.
    start_inference()
    with _pool_lock:
        if _pool is None:
            raise RuntimeError("Inference pool is stopped.")
        future = _pool.submit(_generate, messages, max_new_tokens)
    return future.result()
