# Architecture Notes: Knowledge Base + Internet Research Agent

## Pipeline

```text
Question -> Router (internal vs external vs both) -> Retrieve -> Synthesize with Citations -> Answer
```

## Components

- Private knowledge base search
- External web search
- Source combination logic
- Evidence citation
- Final synthesized response generation

## Design Notes

- Keep provider/model choices swappable behind interfaces (see `multi-llm-router`
  and similar projects in this portfolio for the general pattern).
- Prefer configuration-driven pipelines (YAML/JSON in `configs/`) over hardcoded
  parameters so experiments are reproducible.
