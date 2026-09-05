#!/usr/bin/env python
"""CLI entrypoint for issue #1: ingest a directory of documents into the
configured vector store.

Usage:
    python scripts/ingest_docs.py --config configs/default.yaml
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml

from kb.embeddings import HashingEmbedder
from kb.ingest import IngestionPipeline
from kb.vector_store import JSONVectorStore


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config_path = Path(args.config)
    if not config_path.exists():
        logging.error("Config file not found: %s", config_path)
        return 1
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    embedder = HashingEmbedder(dimensions=config.get("embedding_dimensions", 256))
    store = JSONVectorStore(config["vector_store_path"])
    pipeline = IngestionPipeline(
        embedder=embedder,
        store=store,
        chunk_size=config.get("chunk_size", 800),
        chunk_overlap=config.get("chunk_overlap", 100),
    )

    written = pipeline.ingest_directory(config["source_dir"])
    print(f"Wrote/updated {written} chunks -> {config['vector_store_path']} ({len(store)} total chunks stored)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
