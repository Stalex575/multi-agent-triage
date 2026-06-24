# Multi-Agent Triage and Routing System

This project is a modular LangGraph implementation of a production-style multi-agent triage system. It uses Google's Gemini models through `langchain-google-genai`, Pinecone for semantic cache and retrieval context, and strict `.env`-based secrets loading.

## Architecture

The graph uses these nodes:

1. `Semantic Cache`: embeds the incoming query with `gemini-embedding-2`, searches the Pinecone cache namespace, and returns immediately when the top score is above `0.92`.
2. `Triage Lead`: uses `gemini-3.1-flash-lite` with structured output to classify the domain as exactly `DB_INFRA` or `CLOUD_DEVOPS`.
3. `Database Architect Agent`: retrieves DB context from Pinecone and drafts database tuning, indexing, migration, or query optimization guidance using `gemini-3.5-flash`.
4. `DevOps Agent`: retrieves deployment, networking, CI/CD, Kubernetes, or cloud operations context using `gemini-3.5-flash`.
5. `Integration QA Lead`: checks that the draft answers the query and contains executable configuration, commands, or code. Failed drafts loop back to the selected specialist for at most two retries.
6. `Cache Asynchronous Write`: writes successful novel answers back to Pinecone for later cache hits.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
cp .env.example .env
```

Edit `.env` and set real credentials.

Required `.env` variables:

- `GOOGLE_API_KEY`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `PINECONE_CLOUD`
- `PINECONE_REGION`
- `PINECONE_METRIC`
- `PINECONE_CACHE_NAMESPACE`
- `PINECONE_DB_NAMESPACE`
- `PINECONE_DEVOPS_NAMESPACE`
- `GEMINI_HEAVY_MODEL`
- `GEMINI_FAST_MODEL`
- `GEMINI_EMBEDDING_MODEL`
- `GEMINI_EMBEDDING_DIMENSION`
- `SEMANTIC_CACHE_THRESHOLD`
- `MAX_QA_RETRIES`
- `SPECIALIST_TOP_K`
- `LLM_TIMEOUT_SECONDS`
- `LLM_MAX_RETRIES`
- `CACHE_TEXT_MAX_CHARS`

## Pinecone Index Schema

Create a standard dense vector index, not an integrated embedding index:

- Vector type: dense
- Dimension: `3072` for default `gemini-embedding-2`
- Metric: `cosine`
- Suggested namespaces:
  - `semantic-cache`
  - `db-infra-context`
  - `cloud-devops-context`

Run:

```bash
python scripts/create_pinecone_index.py
python scripts/seed_context.py
```

`gemini-embedding-2` supports reduced dimensions such as `768` and `1536`, but the Pinecone index dimension must exactly match `GEMINI_EMBEDDING_DIMENSION`.

## Run

```bash
python -m triage_router.cli "Postgres writes are spiking latency after a new index rollout. What should I check?"
```

Or from Python:

```python
import asyncio
from triage_router.graph import build_app
from triage_router.state import make_initial_state

async def run() -> None:
    app = build_app()
    result = await app.ainvoke(make_initial_state("How do I harden a blue-green Kubernetes deployment?"))
    print(result["final_response"])

asyncio.run(run())
```

## Tracing with LangSmith

To trace the multi-agent graph execution in LangSmith, export the following environment variables in your terminal before running:

```bash
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="your-langsmith-api-key"
export LANGCHAIN_PROJECT="triage-router-agent"  # Optional
```

Once exported, LangGraph will automatically trace and log all model calls, state changes, and database lookups to your LangSmith dashboard without requiring code changes.

## Notes

- Secrets are read from `.env` only. The application does not hardcode API keys.
- Cache entries are idempotent: query text is hashed into a stable Pinecone vector ID.
- The QA retry loop is deterministic: failed drafts route back to the same specialist until `MAX_QA_RETRIES` is reached.
- Pinecone metadata has practical size limits, so `CACHE_TEXT_MAX_CHARS` caps cached response text. In a larger production system, replace this with an object store pointer.
