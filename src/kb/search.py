"""kb_search tool (issue #2, Phase 1): wraps the embedder + vector store as
a LangChain-callable tool so the agent can query the private knowledge base
as one of its tools.

Informed by the "Stateful agentic run lifecycle" pattern
(E:\\Projects\\LLM\\knowledge-base\\agentic-loops\\stateful-run-lifecycle.md):
a tool call inside an agent loop is a synchronous request/response the loop
resumes on, so the tool function itself must never raise into that
resume cycle -- an empty KB, a blank query, or a no-match-above-threshold
query all resolve to an empty/informative result instead of an exception.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .embeddings import EmbeddingProvider
from .vector_store import VectorStore


@dataclass
class KBSearchResult:
    document: str
    score: float
    metadata: dict


class KBSearcher:
    """Embeds a free-text query and retrieves the top-k most similar chunks
    from the configured vector store, filtered by a minimum similarity
    score. `top_k` and `score_threshold` are configurable both at
    construction time (defaults) and per-call (override)."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        store: VectorStore,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not -1.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be between -1.0 and 1.0")
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.score_threshold = score_threshold

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[KBSearchResult]:
        """Returns ranked matches for `query`. Never raises for an empty
        store or an unmatched query -- both simply produce an empty list."""
        query = (query or "").strip()
        if not query:
            return []
        if len(self.store) == 0:
            return []

        k = top_k if top_k is not None else self.top_k
        threshold = score_threshold if score_threshold is not None else self.score_threshold
        if k <= 0:
            raise ValueError("top_k must be positive")

        [embedding] = self.embedder.embed([query])
        scored = self.store.query_with_scores(embedding, top_k=k)
        return [
            KBSearchResult(document=record.document, score=score, metadata=dict(record.metadata))
            for score, record in scored
            if score >= threshold
        ]


class KBSearchInput(BaseModel):
    query: str = Field(description="Natural-language question to search the private knowledge base for.")


def _format_results(results: List[KBSearchResult]) -> str:
    if not results:
        return "No matching knowledge base results found."
    lines = []
    for i, result in enumerate(results, start=1):
        source = result.metadata.get("source", "unknown")
        lines.append(f"[{i}] (score={result.score:.3f}, source={source}) {result.document}")
    return "\n".join(lines)


def build_kb_search_tool(searcher: KBSearcher) -> StructuredTool:
    """Registers `searcher` as a LangChain `kb_search` StructuredTool."""

    def _run(query: str) -> str:
        try:
            results = searcher.search(query)
        except Exception as exc:  # tool calls must never raise into the agent loop
            return f"kb_search error: {exc}"
        return _format_results(results)

    return StructuredTool.from_function(
        func=_run,
        name="kb_search",
        description=(
            "Search the private knowledge base for chunks relevant to a natural-language "
            "question. Returns the top-k matches ranked by similarity, each with its source."
        ),
        args_schema=KBSearchInput,
    )
