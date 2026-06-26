"""Tests for the FastAPI triage router API."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from triage_router.api import app


@pytest.fixture()
def _mock_lifespan():
    """Bypass the real lifespan so tests don't need live credentials.

    Patches ``Settings.load`` and ``build_app`` to inject lightweight
    fakes into ``app.state`` instead.
    """

    fake_graph = AsyncMock()
    fake_graph.ainvoke.return_value = {
        "final_response": "Mocked triage answer.",
        "qa_passed": True,
    }

    with (
        patch("triage_router.api.Settings.load") as mock_settings,
        patch("triage_router.api.build_app", return_value=fake_graph) as mock_build,
    ):
        mock_settings.return_value = object()  # settings stub
        yield {"graph": fake_graph, "settings": mock_settings, "build_app": mock_build}


@pytest.fixture()
def client(_mock_lifespan) -> TestClient:
    """A ``TestClient`` wired to the app with the mocked lifespan."""

    with TestClient(app) as c:
        yield c


def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_triage_returns_200_with_valid_query(client: TestClient) -> None:
    response = client.post(
        "/api/triage",
        json={"query": "How do I scale a Fargate task?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "How do I scale a Fargate task?"
    assert data["response"] == "Mocked triage answer."
    assert "elapsed_seconds" in data


def test_triage_invokes_graph_with_initial_state(
    client: TestClient, _mock_lifespan: dict
) -> None:
    client.post("/api/triage", json={"query": "Test query"})
    graph = _mock_lifespan["graph"]
    graph.ainvoke.assert_called_once()

    invoked_state = graph.ainvoke.call_args[0][0]
    assert invoked_state["query"] == "Test query"
    assert invoked_state["cache_hit"] is False


def test_triage_rejects_empty_query(client: TestClient) -> None:
    response = client.post("/api/triage", json={"query": ""})
    assert response.status_code == 422


def test_triage_rejects_missing_query(client: TestClient) -> None:
    response = client.post("/api/triage", json={})
    assert response.status_code == 422


def test_triage_rejects_whitespace_only_query(client: TestClient) -> None:
    response = client.post("/api/triage", json={"query": "   "})
    # Pydantic min_length=1 applies after stripping — but FastAPI doesn't
    # strip by default, so "   " has length 3 which passes min_length.
    # However make_initial_state raises ValueError on blank strings.
    assert response.status_code == 422


def test_triage_returns_500_on_graph_failure(
    client: TestClient, _mock_lifespan: dict
) -> None:
    graph = _mock_lifespan["graph"]
    graph.ainvoke.side_effect = RuntimeError("Pinecone connection timeout")

    response = client.post(
        "/api/triage",
        json={"query": "This should fail."},
    )
    assert response.status_code == 500
    assert "Pinecone connection timeout" in response.json()["detail"]
