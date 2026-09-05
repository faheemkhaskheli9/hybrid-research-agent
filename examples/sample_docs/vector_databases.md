# Vector Databases

A vector database stores embeddings — dense numerical representations of
text, images, or other content — and supports approximate nearest-neighbor
search over them. This lets applications retrieve items by semantic
similarity rather than exact keyword match.

Common use cases include retrieval-augmented generation (RAG), semantic
search, recommendation systems, and deduplication. Popular implementations
include Chroma, Pinecone, Weaviate, and pgvector (a PostgreSQL extension).

When choosing a vector store, consider: does it need to run embedded in the
application process or as a separate service, how large is the corpus, and
what metadata filtering does the query pattern require alongside similarity
search.
