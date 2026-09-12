# Knowledge Base + Internet Research Agent

> LLM, RAG & Agentic AI portfolio project — independent open-source implementation.
> This is an original, from-scratch build. It is not affiliated with, and does not
> contain any code, prompts, data, or business logic from, any employer or client.

![status](https://img.shields.io/badge/status-in--progress-yellow)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

## 1. Problem

Some questions are answerable from internal knowledge, others need the open web. An agent needs to decide when to use which, and cite its sources.

## 2. Architecture

```text
Question -> Router (internal vs external vs both) -> Retrieve -> Synthesize with Citations -> Answer
```

## 3. Technology Stack

- Python
- LangChain agents
- Vector DB (private KB)
- Web search API
- OpenAI API

## 4. Feature List

- Private knowledge base search
- External web search
- Source combination logic
- Evidence citation
- Final synthesized response generation

## 5. Implementation Plan

1. Phase 1: Private KB retrieval tool
2. Phase 2: Web search tool integration
3. Phase 3: Routing/decision logic between sources
4. Phase 4: Citation-aware synthesis

## Task Tracking

Work is broken into phase-tagged user stories tracked as GitHub Issues, not in this file. To see what's open:

    gh issue list --repo faheemkhaskheli9/hybrid-research-agent --state open --label type:user-story

Implement Phase 1 issues first (later phases depend on it). When you start one, add label `status:in-progress`. When you finish, close it referencing the commit (e.g. `git commit -m "... Closes #4"`) and push.

## 6. Repository Structure

```text
hybrid-research-agent/
├── README.md
├── LICENSE
├── .gitignore
├── pyproject.toml
├── .env.example
├── docker/
├── docs/
│   ├── architecture.md
│   └── evaluation.md
├── src/
├── tests/
├── configs/
├── scripts/
├── notebooks/
├── examples/
├── assets/
└── .github/
    └── workflows/
```

## 7. Setup

```bash
git clone <this-repo-url>
cd hybrid-research-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # or: pip install -e .
cp .env.example .env              # fill in API keys / config
python scripts/ingest_docs.py --config configs/default.yaml
```

## 8. Dataset

Phase 1 ingests the sample Markdown docs in `examples/sample_docs/` (short,
original explainers on vector DBs and RAG — no proprietary or
employer-owned content). Point `configs/default.yaml`'s `source_dir` at any
directory of `.txt`/`.md` files to ingest your own. Embeddings default to a
deterministic offline `HashingEmbedder` (see `src/kb/embeddings.py`) so
ingestion and tests need no API key; `OpenAIEmbedder` is provided as a
drop-in real-provider swap.

## 9. Training / Execution

Ingest a directory of documents into the configured vector store:

```bash
python scripts/ingest_docs.py --config configs/default.yaml
```

`configs/default.yaml` names the embedding provider/model and vector store
backend as data (`src/kb/config.py`), not code:

```yaml
embedding:
  provider: hashing   # hashing (offline, default) | openai (needs OPENAI_API_KEY)
  model: text-embedding-3-small
  dimensions: 256
vector_store:
  backend: json        # local, dependency-free (see src/kb/vector_store.py)
  path: examples/vector_store.json
```

Switching `embedding.provider` to `openai` (with `OPENAI_API_KEY` set — see
`.env.example`) and re-running the same command swaps embedding models with
no code change. An invalid config (unknown provider/backend, missing API
key, out-of-range chunk size) raises `KBConfigError` with a message naming
the bad field before ingestion starts.

## 10. Evaluation

Document evaluation metrics and how to reproduce them here (see `docs/evaluation.md`).

## 11. Results

_To be filled in as the implementation progresses — screenshots, metrics tables, and
sample outputs go here._

## 12. API

_If this project exposes an API, document the main endpoints here (or link to
auto-generated OpenAPI docs, e.g. `/docs` for FastAPI)._

## 13. Docker

```bash
docker build -t hybrid-research-agent .
docker run -p 8000:8000 hybrid-research-agent
```

## 14. Tests

```bash
pytest tests/
```

## 15. Limitations

- This is a from-scratch, independent recreation built for portfolio purposes.
- Performance numbers, once added, are based on public datasets and are not
  representative of any production system's real-world results.

## 16. Future Work

- Expand evaluation coverage and add CI-based regression checks.
- Add more configuration presets and deployment targets.
- Track open items as GitHub Issues.

## 17. Disclosure

This repository is an **independent open-source recreation inspired by the kind of
production systems I have worked on professionally**. It contains no employer or
client source code, prompts, datasets, credentials, architecture diagrams, or
business logic. All code, data, and documentation here are original or built on
publicly available datasets and open-source tools.

---
_Last updated: 2026-08-18_
