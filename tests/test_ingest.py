"""Regression tests for issue #1: end-to-end ingestion with idempotency."""
from kb.embeddings import HashingEmbedder
from kb.ingest import IngestionPipeline, iter_source_files
from kb.vector_store import JSONVectorStore


def _write_docs(dir_path):
    (dir_path / "a.txt").write_text("Alpha document about vector search and embeddings.", encoding="utf-8")
    (dir_path / "b.md").write_text("Beta document about retrieval augmented generation.", encoding="utf-8")
    (dir_path / "ignored.pdf").write_bytes(b"%PDF-not-supported-yet")


def test_ingest_directory_writes_chunks_with_metadata(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    _write_docs(docs_dir)

    store = JSONVectorStore(tmp_path / "store.json")
    pipeline = IngestionPipeline(embedder=HashingEmbedder(dimensions=32), store=store, chunk_size=1000, chunk_overlap=0)

    written = pipeline.ingest_directory(docs_dir)
    assert written == 2  # one chunk per .txt/.md file at this chunk size
    assert len(store) == 2

    sources = {record.metadata["source"] for record in store._records.values()}
    assert str(docs_dir / "a.txt") in sources
    assert str(docs_dir / "b.md") in sources
    for record in store._records.values():
        assert "title" in record.metadata
        assert "chunk_index" in record.metadata


def test_reingesting_same_directory_is_idempotent(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    _write_docs(docs_dir)

    store = JSONVectorStore(tmp_path / "store.json")
    pipeline = IngestionPipeline(embedder=HashingEmbedder(dimensions=32), store=store, chunk_size=1000, chunk_overlap=0)

    pipeline.ingest_directory(docs_dir)
    count_after_first = len(store)

    pipeline.ingest_directory(docs_dir)  # re-run on identical source set
    assert len(store) == count_after_first  # no duplicates


def test_unsupported_extensions_are_skipped(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    _write_docs(docs_dir)

    files = iter_source_files(docs_dir)
    assert all(f.suffix in {".txt", ".md"} for f in files)
    assert len(files) == 2


def test_missing_source_dir_raises(tmp_path):
    store = JSONVectorStore(tmp_path / "store.json")
    pipeline = IngestionPipeline(embedder=HashingEmbedder(dimensions=32), store=store)
    try:
        pipeline.ingest_directory(tmp_path / "does_not_exist")
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass
