"""Vector store interface (swappable — a real deployment would point this
at Chroma/pgvector/Pinecone; `JSONVectorStore` is a dependency-free local
implementation so this phase is runnable without standing up external
infrastructure).
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class VectorRecord:
    id: str
    embedding: List[float]
    document: str
    metadata: Dict = field(default_factory=dict)


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, records: List[VectorRecord]) -> None:
        """Insert or replace records by id — re-ingesting the same id must
        not create a duplicate (see issue #1 idempotency criterion)."""
        raise NotImplementedError

    @abstractmethod
    def query(self, embedding: np.ndarray, top_k: int = 5) -> List[VectorRecord]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_ids(self) -> set:
        raise NotImplementedError


class JSONVectorStore(VectorStore):
    """Persists records as one JSON file. Fine for a portfolio-scale local
    knowledge base; not intended for production-scale corpora."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._records: Dict[str, VectorRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        for item in raw:
            record = VectorRecord(**item)
            self._records[record.id] = record

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump([vars(r) for r in self._records.values()], fh)
            tmp_path.replace(self.path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    def upsert(self, records: List[VectorRecord]) -> None:
        for record in records:
            self._records[record.id] = record  # upsert by id -> idempotent
        self._save()

    def query(self, embedding: np.ndarray, top_k: int = 5) -> List[VectorRecord]:
        if not self._records:
            return []
        scored = []
        query_vec = np.asarray(embedding, dtype=np.float64)
        for record in self._records.values():
            vec = np.asarray(record.embedding, dtype=np.float64)
            denom = (np.linalg.norm(query_vec) * np.linalg.norm(vec)) or 1.0
            score = float(np.dot(query_vec, vec) / denom)
            scored.append((score, record))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [record for _, record in scored[:top_k]]

    def __len__(self) -> int:
        return len(self._records)

    def get_ids(self) -> set:
        return set(self._records.keys())
