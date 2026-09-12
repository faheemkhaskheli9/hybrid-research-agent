#!/usr/bin/env python
"""CLI entrypoint for issue #1/#3: ingest a directory of documents into the
configured vector store, using a config-driven embedding provider and
vector store backend (see src/kb/config.py).

Usage:
    python scripts/ingest_docs.py --config configs/default.yaml
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from kb.config import KBConfigError, build_embedder, build_vector_store, load_kb_config
from kb.ingest import IngestionPipeline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        config = load_kb_config(args.config)
        embedder = build_embedder(config.embedding)
        store = build_vector_store(config.vector_store)
    except KBConfigError as exc:
        logging.error("%s", exc)
        return 1

    pipeline = IngestionPipeline(
        embedder=embedder,
        store=store,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    written = pipeline.ingest_directory(config.source_dir)
    print(
        f"Wrote/updated {written} chunks -> {config.vector_store.path} "
        f"({len(store)} total chunks stored, embedding.provider={config.embedding.provider})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
