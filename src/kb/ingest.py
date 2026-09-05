"""Ingestion pipeline (issue #1, Phase 1): reads documents from a directory,
chunks them, embeds each chunk, and upserts into the configured vector
store with source metadata (path, title, chunk index) for later citation.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterable, List

from .chunking import chunk_text
from .embeddings import EmbeddingProvider
from .vector_store import VectorRecord, VectorStore

logger = logging.getLogger(__name__)

DEFAULT_EXTENSIONS = {".txt", ".md"}


def _chunk_id(source: str, chunk_index: int, chunk: str) -> str:
    """Deterministic id derived from source path + chunk index + chunk
    content, so re-ingesting the same source produces the same ids
    (idempotent upsert) but an edited chunk gets a new id rather than
    silently overwriting the old text under a stale key."""
    digest = hashlib.sha256(f"{source}::{chunk_index}::{chunk}".encode("utf-8")).hexdigest()
    return digest


def iter_source_files(source_dir: str | Path, extensions: Iterable[str] = DEFAULT_EXTENSIONS) -> List[Path]:
    source_dir = Path(source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    return sorted(p for p in source_dir.rglob("*") if p.is_file() and p.suffix.lower() in extensions)


class IngestionPipeline:
    def __init__(self, embedder: EmbeddingProvider, store: VectorStore, chunk_size: int = 800, chunk_overlap: int = 100):
        self.embedder = embedder
        self.store = store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_file(self, path: Path) -> int:
        """Ingests one file; returns the number of new/updated chunks."""
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        if not chunks:
            logger.info("No content chunks extracted from %s", path)
            return 0

        embeddings = self.embedder.embed(chunks)
        records = [
            VectorRecord(
                id=_chunk_id(str(path), i, chunk),
                embedding=embedding.tolist(),
                document=chunk,
                metadata={"source": str(path), "title": path.stem, "chunk_index": i},
            )
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]
        self.store.upsert(records)
        logger.info("Ingested %s: %d chunks", path, len(records))
        return len(records)

    def ingest_directory(self, source_dir: str | Path) -> int:
        total = 0
        for path in iter_source_files(source_dir):
            total += self.ingest_file(path)
        logger.info("Ingestion run complete: %d chunks written from %s", total, source_dir)
        return total
