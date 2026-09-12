"""Config-driven embedding model and vector store selection (issue #3).

``configs/default.yaml`` names the embedding provider/model and the vector
store backend/connection settings as data, not code — swapping from the
local hashing embedder to OpenAI embeddings (or, later, a different vector
DB backend) is a config edit plus re-running ingestion, never a source
change. ``load_kb_config`` validates the parsed YAML at startup: an unknown
provider/backend or a missing required field (e.g. no ``OPENAI_API_KEY`` when
``embedding.provider: openai``) raises :class:`KBConfigError` with a message
naming the bad field, instead of failing confusingly deep inside ingestion.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .embeddings import EmbeddingProvider, HashingEmbedder, OpenAIEmbedder
from .vector_store import JSONVectorStore, VectorStore

_EMBEDDING_PROVIDERS = {"hashing", "openai"}
_VECTOR_STORE_BACKENDS = {"json"}


class KBConfigError(ValueError):
    """The KB config file is missing, unparseable, or has an invalid value."""


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "hashing"
    model: str = "text-embedding-3-small"  # only meaningful for provider: openai
    dimensions: int = 256

    def __post_init__(self) -> None:
        if self.provider not in _EMBEDDING_PROVIDERS:
            raise KBConfigError(
                f"embedding.provider must be one of {sorted(_EMBEDDING_PROVIDERS)}, "
                f"got {self.provider!r}"
            )
        if self.dimensions <= 0:
            raise KBConfigError(f"embedding.dimensions must be positive, got {self.dimensions!r}")


@dataclass(frozen=True)
class VectorStoreConfig:
    backend: str = "json"
    path: str = "examples/vector_store.json"

    def __post_init__(self) -> None:
        if self.backend not in _VECTOR_STORE_BACKENDS:
            raise KBConfigError(
                f"vector_store.backend must be one of {sorted(_VECTOR_STORE_BACKENDS)}, "
                f"got {self.backend!r}"
            )
        if not str(self.path).strip():
            raise KBConfigError("vector_store.path must not be empty")


@dataclass(frozen=True)
class KBConfig:
    source_dir: str
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    chunk_size: int = 800
    chunk_overlap: int = 100

    def __post_init__(self) -> None:
        if not str(self.source_dir).strip():
            raise KBConfigError("source_dir must not be empty")
        if self.chunk_size <= 0:
            raise KBConfigError(f"chunk_size must be positive, got {self.chunk_size!r}")
        if self.chunk_overlap < 0:
            raise KBConfigError(f"chunk_overlap must be >= 0, got {self.chunk_overlap!r}")


def _require_dict(value: Any, field: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise KBConfigError(f"{field} must be a mapping, got {value!r}")
    return value


def load_kb_config(path: str | Path, *, env: dict | None = None) -> KBConfig:
    """Parse and validate a KB config file.

    ``env`` defaults to ``os.environ`` and is only consulted (for
    ``OPENAI_API_KEY``) when ``embedding.provider: openai`` — overridable so
    callers/tests don't need to mutate real process environment.
    """
    env = os.environ if env is None else env
    path = Path(path)
    if not path.is_file():
        raise KBConfigError(f"KB config file not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise KBConfigError(f"{path.name} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise KBConfigError(f"{path.name} must contain a YAML mapping at the top level")

    try:
        # Legacy flat keys (embedding_dimensions) are accepted as defaults for
        # the nested embedding block so existing configs keep working.
        embedding_raw = _require_dict(raw.get("embedding"), "embedding")
        embedding_raw.setdefault("dimensions", raw.get("embedding_dimensions", 256))
        embedding = EmbeddingConfig(**embedding_raw)

        vector_store_raw = _require_dict(raw.get("vector_store"), "vector_store")
        vector_store_raw.setdefault("path", raw.get("vector_store_path", "examples/vector_store.json"))
        vector_store = VectorStoreConfig(**vector_store_raw)
    except TypeError as exc:
        raise KBConfigError(f"invalid field in {path.name}: {exc}") from exc

    if embedding.provider == "openai" and not env.get("OPENAI_API_KEY"):
        raise KBConfigError(
            "embedding.provider is 'openai' but OPENAI_API_KEY is not set "
            "(see .env.example) — set it before running ingestion"
        )

    try:
        return KBConfig(
            source_dir=raw.get("source_dir", "examples/sample_docs"),
            embedding=embedding,
            vector_store=vector_store,
            chunk_size=raw.get("chunk_size", 800),
            chunk_overlap=raw.get("chunk_overlap", 100),
        )
    except TypeError as exc:
        raise KBConfigError(f"invalid field in {path.name}: {exc}") from exc


def build_embedder(config: EmbeddingConfig) -> EmbeddingProvider:
    if config.provider == "hashing":
        return HashingEmbedder(dimensions=config.dimensions)
    if config.provider == "openai":
        return OpenAIEmbedder(model=config.model, dimensions=config.dimensions)
    raise KBConfigError(f"unknown embedding provider: {config.provider!r}")  # pragma: no cover


def build_vector_store(config: VectorStoreConfig) -> VectorStore:
    if config.backend == "json":
        return JSONVectorStore(config.path)
    raise KBConfigError(f"unknown vector store backend: {config.backend!r}")  # pragma: no cover
