"""Tests for config-driven embedding model / vector store selection (issue #3)."""
import pytest

from kb.config import (
    KBConfigError,
    build_embedder,
    build_vector_store,
    load_kb_config,
)
from kb.embeddings import HashingEmbedder, OpenAIEmbedder
from kb.vector_store import JSONVectorStore


def _write(tmp_path, text, name="config.yaml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


HASHING_CONFIG = """
source_dir: examples/sample_docs
chunk_size: 500
chunk_overlap: 50
embedding:
  provider: hashing
  dimensions: 128
vector_store:
  backend: json
  path: examples/vector_store.json
"""


def test_valid_hashing_config_loads(tmp_path):
    config = load_kb_config(_write(tmp_path, HASHING_CONFIG))
    assert config.embedding.provider == "hashing"
    assert config.embedding.dimensions == 128
    assert config.vector_store.backend == "json"
    assert config.chunk_size == 500


def test_legacy_flat_keys_still_work(tmp_path):
    legacy = "source_dir: docs\nvector_store_path: store.json\nembedding_dimensions: 64\n"
    config = load_kb_config(_write(tmp_path, legacy))
    assert config.embedding.dimensions == 64
    assert config.vector_store.path == "store.json"


def test_openai_provider_requires_api_key(tmp_path):
    config_text = HASHING_CONFIG.replace("provider: hashing", "provider: openai")
    with pytest.raises(KBConfigError, match="OPENAI_API_KEY"):
        load_kb_config(_write(tmp_path, config_text), env={})


def test_openai_provider_succeeds_with_api_key(tmp_path):
    config_text = HASHING_CONFIG.replace("provider: hashing", "provider: openai")
    config = load_kb_config(_write(tmp_path, config_text), env={"OPENAI_API_KEY": "sk-fake"})
    assert config.embedding.provider == "openai"


def test_unknown_embedding_provider_rejected(tmp_path):
    config_text = HASHING_CONFIG.replace("provider: hashing", "provider: bogus")
    with pytest.raises(KBConfigError, match="embedding.provider"):
        load_kb_config(_write(tmp_path, config_text))


def test_unknown_vector_store_backend_rejected(tmp_path):
    config_text = HASHING_CONFIG.replace("backend: json", "backend: pinecone")
    with pytest.raises(KBConfigError, match="vector_store.backend"):
        load_kb_config(_write(tmp_path, config_text))


def test_missing_file_raises(tmp_path):
    with pytest.raises(KBConfigError, match="not found"):
        load_kb_config(tmp_path / "nope.yaml")


def test_bad_yaml_reports_clearly(tmp_path):
    with pytest.raises(KBConfigError, match="not valid YAML"):
        load_kb_config(_write(tmp_path, "embedding: [1, 2\n"))


def test_non_positive_chunk_size_rejected(tmp_path):
    config_text = HASHING_CONFIG.replace("chunk_size: 500", "chunk_size: 0")
    with pytest.raises(KBConfigError, match="chunk_size"):
        load_kb_config(_write(tmp_path, config_text))


def test_negative_dimensions_rejected(tmp_path):
    config_text = HASHING_CONFIG.replace("dimensions: 128", "dimensions: -1")
    with pytest.raises(KBConfigError, match="dimensions"):
        load_kb_config(_write(tmp_path, config_text))


def test_build_embedder_returns_hashing_embedder(tmp_path):
    config = load_kb_config(_write(tmp_path, HASHING_CONFIG))
    embedder = build_embedder(config.embedding)
    assert isinstance(embedder, HashingEmbedder)
    assert embedder.dimensions == 128


def test_build_embedder_returns_openai_embedder(tmp_path):
    config_text = HASHING_CONFIG.replace("provider: hashing", "provider: openai")
    config = load_kb_config(_write(tmp_path, config_text), env={"OPENAI_API_KEY": "sk-fake"})
    embedder = build_embedder(config.embedding)
    assert isinstance(embedder, OpenAIEmbedder)


def test_build_vector_store_returns_json_store(tmp_path):
    config_text = HASHING_CONFIG.replace(
        "path: examples/vector_store.json", f"path: {tmp_path / 'store.json'}"
    )
    config = load_kb_config(_write(tmp_path, config_text))
    store = build_vector_store(config.vector_store)
    assert isinstance(store, JSONVectorStore)


def test_swapping_provider_in_config_produces_different_embedder_with_no_code_change(tmp_path):
    """Simulates the acceptance criterion: change config, re-run, no code edit."""
    hashing_config = load_kb_config(_write(tmp_path, HASHING_CONFIG))
    openai_config = load_kb_config(
        _write(tmp_path, HASHING_CONFIG.replace("provider: hashing", "provider: openai"), "o.yaml"),
        env={"OPENAI_API_KEY": "sk-fake"},
    )

    hashing_embedder = build_embedder(hashing_config.embedding)
    openai_embedder = build_embedder(openai_config.embedding)
    assert type(hashing_embedder) is not type(openai_embedder)
