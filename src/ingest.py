"""Rebuild the PDF search index. Run from project root: python src/ingest.py"""

from pathlib import Path
import warnings

import pymupdf
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

if __package__:
    from .config import (CHROMA_PATH, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME,
                         DATA_PATH, EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL)
else:
    from config import (CHROMA_PATH, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME,
                        DATA_PATH, EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL)


def ensure_index():
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    """Build on first startup; reuse a compatible, nonempty persisted index."""
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    if COLLECTION_NAME in [collection.name for collection in client.list_collections()]:
        collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
        if collection.count() and (collection.metadata or {}).get("embedding_model") == EMBEDDING_MODEL:
            return
    ingest()

def extract_pages(pdf_path: Path) -> list[dict]:
    """Return nonempty pages with 1-based PDF page numbers."""
    pages = []
    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text", sort=True).strip()
            if not text:
                warnings.warn(f"{pdf_path.name}, page {page_number}: no text; skipped.")
                continue
            pages.append({
                "text": text,
                "metadata": {
                    "company": pdf_path.stem,
                    "source": pdf_path.name,
                    "page": page_number,
                },
            })
    if not pages:
        raise ValueError(f"No extractable text in {pdf_path.name}; OCR may be needed.")
    return pages


def split_text(text, tokenizer, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split by token positions, retaining original spelling and formatting."""
    if not 0 <= overlap < chunk_size:
        raise ValueError("Require 0 <= overlap < chunk_size.")

    # Character offsets let us slice the original text instead of decoding tokens.
    offsets = tokenizer(
        text, add_special_tokens=False, return_offsets_mapping=True,
        truncation=False, verbose=False,
    )["offset_mapping"]
    chunks = []
    for start in range(0, len(offsets), chunk_size - overlap):
        end = min(start + chunk_size, len(offsets))
        chunk = text[offsets[start][0]:offsets[end - 1][1]].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(offsets):
            break
    return chunks


def build_chunks(data_path: Path = DATA_PATH) -> list[dict]:
    """Prepare records for later embedding and ChromaDB storage."""
    pdf_paths = sorted(data_path.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {data_path}")

    # Downloads only tokenizer files and caches them automatically.
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL, use_fast=True)
    chunks = []
    for pdf_path in pdf_paths:
        for page in extract_pages(pdf_path):
            for index, text in enumerate(split_text(page["text"], tokenizer), start=1):
                chunks.append({
                    "id": f"{pdf_path.stem}-p{page['metadata']['page']}-c{index}",
                    "text": text,
                    "metadata": page["metadata"].copy(),
                })
    return chunks


def ingest():
    """Embed all chunks, then replace the collection with the current dataset."""
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    chunks = build_chunks()
    for source in sorted({chunk["metadata"]["source"] for chunk in chunks}):
        count = sum(chunk["metadata"]["source"] == source for chunk in chunks)
        print(f"{source}: {count} chunks")
    print(f"Total: {len(chunks)} chunks")

    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    texts = [chunk["text"] for chunk in chunks]
    # Fail rather than silently embed truncated chunks after a config change.
    lengths = model.tokenizer(texts, truncation=False, verbose=False)["input_ids"]
    if any(len(tokens) > model.max_seq_length for tokens in lengths):
        raise ValueError("A chunk exceeds the embedding model's input limit; reduce CHUNK_SIZE.")

    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"Embeddings: {embeddings.shape[0]} vectors, {embeddings.shape[1]} dimensions")

    # Complete extraction and embedding before replacing an existing index.
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    if COLLECTION_NAME in [collection.name for collection in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=None,  # We supply MiniLM embeddings explicitly.
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"embedding_model": EMBEDDING_MODEL},
    )
    batch_size = client.get_max_batch_size()
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        collection.add(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["text"] for chunk in batch],
            metadatas=[chunk["metadata"] for chunk in batch],
            embeddings=embeddings[start:start + batch_size].tolist(),
        )
    if collection.count() != len(chunks):
        raise RuntimeError("Stored chunk count does not match the prepared chunks.")
    print(f"Saved {collection.count()} chunks to {CHROMA_PATH} ({COLLECTION_NAME})")


if __name__ == "__main__":
    ingest()
