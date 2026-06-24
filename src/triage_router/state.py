"""LangGraph state model."""

from __future__ import annotations

from typing import Literal

from typing_extensions import NotRequired, TypedDict


DomainClassification = Literal["DB_INFRA", "CLOUD_DEVOPS"]


class TriageState(TypedDict):
    """State carried across the LangGraph workflow."""

    query: str
    cache_hit: NotRequired[bool]
    domain_classification: NotRequired[DomainClassification]
    retrieved_context: NotRequired[str]
    draft_solution: NotRequired[str]
    final_response: NotRequired[str]
    qa_passed: NotRequired[bool]
    qa_feedback: NotRequired[str]
    retry_count: NotRequired[int]


def make_initial_state(query: str) -> TriageState:
    """Create a complete initial graph state for a user query."""

    cleaned_query = query.strip()
    if not cleaned_query:
        raise ValueError("query must not be empty")

    return {
        "query": cleaned_query,
        "cache_hit": False,
        "domain_classification": "CLOUD_DEVOPS",
        "retrieved_context": "",
        "draft_solution": "",
        "final_response": "",
        "qa_passed": False,
        "qa_feedback": "",
        "retry_count": 0,
    }

