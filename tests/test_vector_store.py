from kb.vector_store import JSONVectorStore, VectorRecord


def _record(id_, value):
    return VectorRecord(id=id_, embedding=[value, 0.0, 0.0], document=f"doc {id_}", metadata={"chunk_index": 0})


def test_upsert_and_persist_round_trip(tmp_path):
    path = tmp_path / "store.json"
    store = JSONVectorStore(path)
    store.upsert([_record("a", 1.0), _record("b", 0.5)])
    assert len(store) == 2

    reloaded = JSONVectorStore(path)
    assert len(reloaded) == 2
    assert reloaded.get_ids() == {"a", "b"}


def test_upsert_same_id_does_not_duplicate(tmp_path):
    store = JSONVectorStore(tmp_path / "store.json")
    store.upsert([_record("a", 1.0)])
    store.upsert([_record("a", 1.0)])  # re-ingest same id
    assert len(store) == 1


def test_query_returns_closest_by_cosine_similarity(tmp_path):
    store = JSONVectorStore(tmp_path / "store.json")
    store.upsert([_record("close", 1.0), _record("far", -1.0)])
    results = store.query([1.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].id == "close"


def test_query_on_empty_store_returns_empty():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        store = JSONVectorStore(f"{d}/store.json")
        assert store.query([1.0, 0.0, 0.0]) == []
