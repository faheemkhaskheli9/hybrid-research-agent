# Retrieval-Augmented Generation

Retrieval-augmented generation (RAG) combines a retrieval step with a
language model's generation step: relevant passages are fetched from an
external knowledge source and included in the model's context so it can
answer using up-to-date or private information it wasn't trained on.

A typical RAG pipeline chunks source documents, embeds each chunk, stores
the embeddings in a vector index, and at query time retrieves the top-K
most similar chunks to inject into the prompt alongside the user's
question. Citing the source of each retrieved chunk lets the final answer
be traced back to specific documents.
