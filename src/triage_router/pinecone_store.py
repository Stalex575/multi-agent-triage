"""Pinecone access layer for cache and domain context retrieval."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from pinecone.grpc import PineconeGRPC as Pinecone

from triage_router.config import Settings
from triage_router.embedding_format import cache_similarity_text, retrieval_query_text
from triage_router.state import DomainClassification


class EmbeddingsClient(Protocol):
    """Subset of LangChain embeddings API used by the store."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""


@dataclass(frozen=True, slots=True)
class CacheLookup:
    """Semantic cache lookup result."""

    hit: bool
    score: float
    final_response: str


class PineconeStore:
    """Thin async wrapper around Pinecone vector operations."""

    def __init__(self, settings: Settings, embeddings: EmbeddingsClient) -> None:
        self._settings = settings
        self._embeddings = embeddings
        self._client = Pinecone(api_key=settings.pinecone_api_key)
        self._index = self._client.Index(settings.pinecone_index_name)

    async def lookup_cache(self, query: str) -> CacheLookup:
        """Return a cached final response when cosine similarity clears threshold."""

        vector = await self._embed(cache_similarity_text(query))
        response = await asyncio.to_thread(
            self._index.query,
            vector=vector,
            top_k=1,
            include_metadata=True,
            include_values=False,
            namespace=self._settings.pinecone_cache_namespace,
        )
        matches = _matches(response)
        if not matches:
            return CacheLookup(hit=False, score=0.0, final_response="")

        best = matches[0]
        score = _score(best)
        metadata = _metadata(best)
        final_response = str(metadata.get("final_response", ""))
        hit = score >= self._settings.semantic_cache_threshold and bool(final_response)
        return CacheLookup(hit=hit, score=score, final_response=final_response if hit else "")

    async def retrieve_context(self, query: str, domain: DomainClassification) -> str:
        """Retrieve domain-specific context from the correct Pinecone namespace."""

        namespace = self._namespace_for_domain(domain)
        vector = await self._embed(retrieval_query_text(query))
        response = await asyncio.to_thread(
            self._index.query,
            vector=vector,
            top_k=self._settings.specialist_top_k,
            include_metadata=True,
            include_values=False,
            namespace=namespace,
        )
        chunks: list[str] = []
        for number, match in enumerate(_matches(response), start=1):
            metadata = _metadata(match)
            title = str(metadata.get("title", f"Context {number}"))
            source = str(metadata.get("source", "unknown"))
            text = str(metadata.get("text", ""))
            if text:
                chunks.append(f"[{number}] {title}\nsource: {source}\n{text}")
        return "\n\n".join(chunks)

    async def write_cache(
        self,
        *,
        query: str,
        domain: DomainClassification,
        final_response: str,
    ) -> str:
        """Write a successful response to the semantic cache namespace."""

        vector = await self._embed(cache_similarity_text(query))
        vector_id = self.cache_id(query)
        metadata = {
            "query": query[:4096],
            "domain": domain,
            "final_response": final_response[: self._settings.cache_text_max_chars],
            "created_at": datetime.now(UTC).isoformat(),
            "model": self._settings.heavy_model,
        }
        await asyncio.to_thread(
            self._index.upsert,
            vectors=[{"id": vector_id, "values": vector, "metadata": metadata}],
            namespace=self._settings.pinecone_cache_namespace,
        )
        return vector_id

    async def upsert_context_documents(
        self,
        *,
        namespace: str,
        documents: list[dict[str, str]],
    ) -> None:
        """Embed and upsert domain context documents for retrieval."""

        vectors: list[dict[str, Any]] = []
        for document in documents:
            title = document["title"]
            text = document["text"]
            source = document.get("source", "seed")
            vector = await self._embed(document["embedding_text"])
            vector_id = _stable_id(f"{namespace}:{title}:{text}")
            vectors.append(
                {
                    "id": vector_id,
                    "values": vector,
                    "metadata": {
                        "title": title,
                        "text": text,
                        "source": source,
                    },
                }
            )

        if vectors:
            await asyncio.to_thread(
                self._index.upsert,
                vectors=vectors,
                namespace=namespace,
            )

    def cache_id(self, query: str) -> str:
        """Return the stable vector ID for a cache query."""

        return f"cache-{_stable_id(query)}"

    def _namespace_for_domain(self, domain: DomainClassification) -> str:
        if domain == "DB_INFRA":
            return self._settings.pinecone_db_namespace
        return self._settings.pinecone_devops_namespace

    async def _embed(self, text: str) -> list[float]:
        return await asyncio.to_thread(self._embeddings.embed_query, text)


def _stable_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _matches(response: Any) -> list[Any]:
    if isinstance(response, dict):
        return list(response.get("matches") or [])
    return list(getattr(response, "matches", []) or [])


def _metadata(match: Any) -> dict[str, Any]:
    if isinstance(match, dict):
        return dict(match.get("metadata") or {})
    return dict(getattr(match, "metadata", {}) or {})


def _score(match: Any) -> float:
    if isinstance(match, dict):
        return float(match.get("score") or 0.0)
    return float(getattr(match, "score", 0.0) or 0.0)

