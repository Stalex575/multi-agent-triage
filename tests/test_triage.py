"""Mocked unit tests for the LangGraph triage architecture.

Every external dependency — ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings,
and the Pinecone gRPC client — is fully mocked so that the test suite passes on
CI runners without live API keys.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from triage_router.config import Settings
from triage_router.graph import (
    _route_after_cache,
    _route_after_qa,
    _route_after_triage,
)
from triage_router.nodes import (
    DomainClassificationResult,
    QAResult,
    TriageNodes,
)
from triage_router.pinecone_store import CacheLookup, PineconeStore
from triage_router.state import TriageState, make_initial_state


# ──────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ──────────────────────────────────────────────────────────────────────────────


def _fake_settings(**overrides: Any) -> Settings:
    """Return a Settings instance with dummy secrets — no .env file needed."""
    defaults = dict(
        google_api_key="fake-google-key",
        pinecone_api_key="fake-pinecone-key",
        pinecone_index_name="test-index",
        pinecone_cloud="aws",
        pinecone_region="us-east-1",
        pinecone_metric="cosine",
        pinecone_cache_namespace="semantic-cache",
        pinecone_db_namespace="db-infra-context",
        pinecone_devops_namespace="cloud-devops-context",
        heavy_model="gemini-3.5-flash",
        fast_model="gemini-3.1-flash-lite",
        embedding_model="gemini-embedding-2",
        embedding_dimension=3072,
        semantic_cache_threshold=0.92,
        max_qa_retries=2,
        specialist_top_k=5,
        llm_timeout_seconds=60,
        llm_max_retries=2,
        cache_text_max_chars=20_000,
    )
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture()
def settings() -> Settings:
    return _fake_settings()


def _fake_embedding_vector(dim: int = 8) -> list[float]:
    """Return a short dummy embedding vector."""
    return [0.1] * dim


def _mock_embeddings_client() -> MagicMock:
    """Return a mock that satisfies the EmbeddingsClient protocol."""
    mock = MagicMock()
    mock.embed_query.return_value = _fake_embedding_vector()
    return mock


# ──────────────────────────────────────────────────────────────────────────────
# Helpers to build TriageNodes with fully mocked internals
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _MockModelBundle:
    """Stand-in for models.ModelBundle with mock LLM clients."""

    router: Any
    specialist: Any
    qa: Any
    embeddings: Any


def _build_mock_nodes(
    settings: Settings,
    *,
    store: PineconeStore | MagicMock | None = None,
    router_result: Any = None,
    specialist_text: str = "mock specialist draft",
    qa_result: QAResult | None = None,
) -> TriageNodes:
    """Construct a TriageNodes object wired to mock services."""

    embeddings = _mock_embeddings_client()

    # --- mock LLM router (with_structured_output chain) ---
    router_chain = AsyncMock()
    if router_result is not None:
        router_chain.ainvoke.return_value = router_result
    router = MagicMock()
    router.with_structured_output.return_value = router_chain

    # --- mock specialist LLM ---
    specialist_response = MagicMock()
    specialist_response.content = specialist_text
    specialist = AsyncMock()
    specialist.ainvoke.return_value = specialist_response

    # --- mock QA LLM (with_structured_output chain) ---
    qa_chain = AsyncMock()
    if qa_result is not None:
        qa_chain.ainvoke.return_value = qa_result
    qa = MagicMock()
    qa.with_structured_output.return_value = qa_chain

    models = _MockModelBundle(
        router=router,
        specialist=specialist,
        qa=qa,
        embeddings=embeddings,
    )

    if store is None:
        store = MagicMock(spec=PineconeStore)
        store.lookup_cache = AsyncMock(
            return_value=CacheLookup(hit=False, score=0.0, final_response="")
        )
        store.retrieve_context = AsyncMock(return_value="mock context")
        store.write_cache = AsyncMock(return_value="cache-abc123")

    return TriageNodes(settings=settings, models=models, store=store)


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Semantic cache hit returns final_response and short-circuits
# ══════════════════════════════════════════════════════════════════════════════


class TestSemanticCacheHit:
    """Verify the graph initialization handles a simulated cache hit."""

    @pytest.mark.asyncio
    async def test_cache_hit_returns_final_response(self, settings: Settings) -> None:
        """When the store reports a cache hit the semantic_cache node should
        propagate cache_hit=True, the cached final_response, and qa_passed=True."""

        cached_answer = "Use READ COMMITTED isolation to avoid lock contention."
        mock_store = MagicMock(spec=PineconeStore)
        mock_store.lookup_cache = AsyncMock(
            return_value=CacheLookup(hit=True, score=0.97, final_response=cached_answer)
        )

        nodes = _build_mock_nodes(settings, store=mock_store)
        state = make_initial_state("Why are writes slow on my Postgres table?")
        result = await nodes.semantic_cache(state)

        assert result["cache_hit"] is True
        assert result["final_response"] == cached_answer
        assert result["qa_passed"] is True

    @pytest.mark.asyncio
    async def test_cache_miss_passes_through(self, settings: Settings) -> None:
        """A cache miss should return cache_hit=False and no final_response."""

        mock_store = MagicMock(spec=PineconeStore)
        mock_store.lookup_cache = AsyncMock(
            return_value=CacheLookup(hit=False, score=0.45, final_response="")
        )

        nodes = _build_mock_nodes(settings, store=mock_store)
        state = make_initial_state("Optimize this Kubernetes deployment")
        result = await nodes.semantic_cache(state)

        assert result["cache_hit"] is False
        assert "final_response" not in result

    def test_route_after_cache_routes_to_end_on_hit(self) -> None:
        """The deterministic _route_after_cache should return 'cache_hit'."""
        state = make_initial_state("query")
        state["cache_hit"] = True
        assert _route_after_cache(state) == "cache_hit"

    def test_route_after_cache_routes_to_triage_on_miss(self) -> None:
        state = make_initial_state("query")
        state["cache_hit"] = False
        assert _route_after_cache(state) == "cache_miss"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Triage Lead correctly classifies domain
# ══════════════════════════════════════════════════════════════════════════════


class TestTriageLeadClassification:
    """Verify the Triage Lead node accurately updates the state domain
    to DB_INFRA or CLOUD_DEVOPS based on mocked LLM outputs."""

    @pytest.mark.asyncio
    async def test_classifies_as_db_infra(self, settings: Settings) -> None:
        """A database-related query should be classified as DB_INFRA."""

        mock_classification = DomainClassificationResult(
            domain_classification="DB_INFRA"
        )
        nodes = _build_mock_nodes(settings, router_result=mock_classification)

        state = make_initial_state("My Postgres VACUUM is not reclaiming dead tuples")
        result = await nodes.triage_lead(state)

        assert result["domain_classification"] == "DB_INFRA"

    @pytest.mark.asyncio
    async def test_classifies_as_cloud_devops(self, settings: Settings) -> None:
        """A DevOps-related query should be classified as CLOUD_DEVOPS."""

        mock_classification = DomainClassificationResult(
            domain_classification="CLOUD_DEVOPS"
        )
        nodes = _build_mock_nodes(settings, router_result=mock_classification)

        state = make_initial_state("Our Kubernetes pods keep getting OOMKilled")
        result = await nodes.triage_lead(state)

        assert result["domain_classification"] == "CLOUD_DEVOPS"

    def test_route_after_triage_forwards_domain(self) -> None:
        """_route_after_triage should return the state's domain_classification."""
        state = make_initial_state("query")
        state["domain_classification"] = "DB_INFRA"
        assert _route_after_triage(state) == "DB_INFRA"

        state["domain_classification"] = "CLOUD_DEVOPS"
        assert _route_after_triage(state) == "CLOUD_DEVOPS"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — QA failure triggers retry loop state modification
# ══════════════════════════════════════════════════════════════════════════════


class TestQARetryLoop:
    """Verify that the routing logic catches a failed QA check and triggers
    a retry loop by modifying the state correctly."""

    @pytest.mark.asyncio
    async def test_qa_failure_sets_retry_state(self, settings: Settings) -> None:
        """When QA fails, the integration_qa_lead node should set qa_passed=False
        and populate qa_feedback for the next retry."""

        qa_result = QAResult(
            qa_passed=False,
            feedback="Missing rollback plan and concrete SQL.",
            final_response="",
        )
        nodes = _build_mock_nodes(settings, qa_result=qa_result)

        state = make_initial_state("Tune this Postgres query")
        state["domain_classification"] = "DB_INFRA"
        state["draft_solution"] = "Just add an index."
        state["retry_count"] = 0

        result = await nodes.integration_qa_lead(state)

        assert result["qa_passed"] is False
        assert "rollback" in result["qa_feedback"].lower()
        assert result["final_response"] == ""

    @pytest.mark.asyncio
    async def test_qa_pass_returns_final_response(self, settings: Settings) -> None:
        """When QA passes, the node should set qa_passed=True and a polished response."""

        polished = "Complete production-grade solution with rollback."
        qa_result = QAResult(
            qa_passed=True,
            feedback="",
            final_response=polished,
        )
        nodes = _build_mock_nodes(settings, qa_result=qa_result)

        state = make_initial_state("Tune this Postgres query")
        state["draft_solution"] = "Detailed solution..."
        result = await nodes.integration_qa_lead(state)

        assert result["qa_passed"] is True
        assert result["final_response"] == polished

    def test_route_after_qa_retries_db_when_under_budget(self) -> None:
        """Before exhausting the retry budget, a failed QA should route back
        to the DB specialist when domain is DB_INFRA."""

        state = make_initial_state("Tune query")
        state["domain_classification"] = "DB_INFRA"
        state["qa_passed"] = False
        state["retry_count"] = 1

        assert _route_after_qa(state, max_qa_retries=2) == "retry_db"

    def test_route_after_qa_retries_devops_when_under_budget(self) -> None:
        """Before exhausting the retry budget, a failed QA should route back
        to the DevOps specialist when domain is CLOUD_DEVOPS."""

        state = make_initial_state("Fix rollout")
        state["domain_classification"] = "CLOUD_DEVOPS"
        state["qa_passed"] = False
        state["retry_count"] = 0

        assert _route_after_qa(state, max_qa_retries=2) == "retry_devops"

    def test_route_after_qa_exhausts_budget(self) -> None:
        """Once retry_count >= max_qa_retries the route should be 'qa_failed_exhausted'."""

        state = make_initial_state("Fix rollout")
        state["domain_classification"] = "CLOUD_DEVOPS"
        state["qa_passed"] = False
        state["retry_count"] = 2

        assert _route_after_qa(state, max_qa_retries=2) == "qa_failed_exhausted"

    @pytest.mark.asyncio
    async def test_qa_exhausted_generates_fallback_response(
        self, settings: Settings
    ) -> None:
        """When QA fails and the retry budget is exhausted the node should
        build a human-readable fallback response with the latest draft and feedback."""

        qa_result = QAResult(
            qa_passed=False,
            feedback="Still missing validation steps.",
            final_response="",
        )
        nodes = _build_mock_nodes(settings, qa_result=qa_result)

        state = make_initial_state("Postgres partitioning problem")
        state["domain_classification"] = "DB_INFRA"
        state["draft_solution"] = "Partition by date range."
        state["retry_count"] = 2  # at budget

        result = await nodes.integration_qa_lead(state)

        assert result["qa_passed"] is False
        assert "QA did not pass" in result["final_response"]
        assert "Partition by date range" in result["final_response"]
        assert "validation steps" in result["final_response"].lower()


# ══════════════════════════════════════════════════════════════════════════════
# TEST — State factory validation
# ══════════════════════════════════════════════════════════════════════════════


class TestStateFactory:
    """Validate the make_initial_state factory used by every graph invocation."""

    def test_creates_valid_initial_state(self) -> None:
        state = make_initial_state("How do I set up replication?")
        assert state["query"] == "How do I set up replication?"
        assert state["cache_hit"] is False
        assert state["retry_count"] == 0

    def test_strips_whitespace(self) -> None:
        state = make_initial_state("  padded query  ")
        assert state["query"] == "padded query"

    def test_rejects_empty_query(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            make_initial_state("   ")
