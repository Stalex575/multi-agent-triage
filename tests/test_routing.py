"""Unit tests for deterministic graph routing helpers."""

from triage_router.graph import _route_after_cache, _route_after_qa, _route_after_triage
from triage_router.state import make_initial_state


def test_cache_hit_routes_to_end_label() -> None:
    state = make_initial_state("Why are writes slow?")
    state["cache_hit"] = True

    assert _route_after_cache(state) == "cache_hit"


def test_cache_miss_routes_to_triage_label() -> None:
    state = make_initial_state("Why are writes slow?")
    state["cache_hit"] = False

    assert _route_after_cache(state) == "cache_miss"


def test_triage_routes_selected_domain() -> None:
    state = make_initial_state("Tune this Postgres query")
    state["domain_classification"] = "DB_INFRA"

    assert _route_after_triage(state) == "DB_INFRA"


def test_qa_failed_routes_back_to_same_specialist_before_budget() -> None:
    state = make_initial_state("Tune this Postgres query")
    state["domain_classification"] = "DB_INFRA"
    state["qa_passed"] = False
    state["retry_count"] = 1

    assert _route_after_qa(state, max_qa_retries=2) == "retry_db"


def test_qa_failed_exhausts_at_budget() -> None:
    state = make_initial_state("Fix this rollout")
    state["domain_classification"] = "CLOUD_DEVOPS"
    state["qa_passed"] = False
    state["retry_count"] = 2

    assert _route_after_qa(state, max_qa_retries=2) == "qa_failed_exhausted"

