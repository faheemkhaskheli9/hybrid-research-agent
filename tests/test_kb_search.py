from kb.embeddings import HashingEmbedder
from kb.search import KBSearcher, build_kb_search_tool
from kb.vector_store import JSONVectorStore, VectorRecord


def _embed(text, dimensions=8):
    return HashingEmbedder(dimensions=dimensions).embed([text])[0].tolist()


def _seed_store(path, docs):
    """docs: list of (id, text) tuples, embedded with an 8-dim hashing embedder."""
    store = JSONVectorStore(path)
    records = [
        VectorRecord(id=doc_id, embedding=_embed(text), document=text, metadata={"source": f"{doc_id}.md"})
        for doc_id, text in docs
    ]
    store.upsert(records)
    return store


def test_search_on_empty_kb_returns_empty_list(tmp_path):
    store = JSONVectorStore(tmp_path / "store.json")
    searcher = KBSearcher(embedder=HashingEmbedder(dimensions=8), store=store)
    assert searcher.search("what is a vector database") == []


def test_search_blank_query_returns_empty_list_without_calling_store(tmp_path):
    store = _seed_store(tmp_path / "store.json", [("a", "vector databases store embeddings")])
    searcher = KBSearcher(embedder=HashingEmbedder(dimensions=8), store=store)
    assert searcher.search("   ") == []


def test_search_returns_ranked_matches_with_metadata(tmp_path):
    store = _seed_store(
        tmp_path / "store.json",
        [
            ("close", "vector databases store embeddings for similarity search"),
            ("far", "sourdough bread requires a long fermentation time"),
        ],
    )
    searcher = KBSearcher(embedder=HashingEmbedder(dimensions=8), store=store, top_k=2)
    results = searcher.search("vector databases store embeddings for similarity search")

    assert len(results) >= 1
    assert results[0].metadata["source"] == "close.md"
    assert results[0].score >= results[-1].score


def test_score_threshold_filters_out_no_match_case(tmp_path):
    store = _seed_store(tmp_path / "store.json", [("far", "sourdough bread requires a long fermentation time")])
    searcher = KBSearcher(embedder=HashingEmbedder(dimensions=8), store=store, score_threshold=0.99)
    # completely unrelated query against a high threshold should yield no matches,
    # not raise, even though the store is non-empty.
    assert searcher.search("quarterly financial report") == []


def test_top_k_and_threshold_are_configurable_per_call(tmp_path):
    store = _seed_store(
        tmp_path / "store.json",
        [("a", "alpha beta gamma"), ("b", "alpha beta delta"), ("c", "alpha beta epsilon")],
    )
    searcher = KBSearcher(embedder=HashingEmbedder(dimensions=8), store=store, top_k=3, score_threshold=0.0)
    default_results = searcher.search("alpha beta gamma")
    limited_results = searcher.search("alpha beta gamma", top_k=1)
    assert len(limited_results) == 1
    assert len(default_results) >= len(limited_results)


def test_invalid_top_k_raises_value_error(tmp_path):
    store = JSONVectorStore(tmp_path / "store.json")
    try:
        KBSearcher(embedder=HashingEmbedder(dimensions=8), store=store, top_k=0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_kb_search_tool_on_empty_kb_does_not_raise(tmp_path):
    store = JSONVectorStore(tmp_path / "store.json")
    searcher = KBSearcher(embedder=HashingEmbedder(dimensions=8), store=store)
    tool = build_kb_search_tool(searcher)

    result = tool.invoke({"query": "anything"})
    assert "No matching knowledge base results found." in result


def test_kb_search_tool_returns_formatted_results_with_source(tmp_path):
    store = _seed_store(tmp_path / "store.json", [("a", "vector databases store embeddings")])
    searcher = KBSearcher(embedder=HashingEmbedder(dimensions=8), store=store)
    tool = build_kb_search_tool(searcher)

    result = tool.invoke({"query": "vector databases store embeddings"})
    assert "a.md" in result
    assert "score=" in result


def test_kb_search_tool_wraps_unexpected_errors_instead_of_raising(tmp_path, monkeypatch):
    store = _seed_store(tmp_path / "store.json", [("a", "vector databases store embeddings")])
    searcher = KBSearcher(embedder=HashingEmbedder(dimensions=8), store=store)

    def _boom(*args, **kwargs):
        raise RuntimeError("embedding backend unavailable")

    monkeypatch.setattr(searcher, "search", _boom)
    tool = build_kb_search_tool(searcher)

    result = tool.invoke({"query": "vector databases"})
    assert "kb_search error" in result
