"""LangGraph workflow assembly."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from triage_router.config import Settings
from triage_router.models import build_models
from triage_router.nodes import TriageNodes
from triage_router.pinecone_store import PineconeStore
from triage_router.state import TriageState


NODE_SEMANTIC_CACHE = "Semantic Cache"
NODE_TRIAGE_LEAD = "Triage Lead"
NODE_DB_AGENT = "Database Architect Agent"
NODE_DEVOPS_AGENT = "DevOps Agent"
NODE_QA_LEAD = "Integration QA Lead"
NODE_CACHE_WRITE = "Cache Asynchronous Write"


def build_app(settings: Settings | None = None):
    """Build and compile the LangGraph workflow."""

    runtime_settings = settings or Settings.from_env_file()
    models = build_models(runtime_settings)
    store = PineconeStore(runtime_settings, models.embeddings)
    nodes = TriageNodes(settings=runtime_settings, models=models, store=store)

    graph = StateGraph(TriageState)
    graph.add_node(NODE_SEMANTIC_CACHE, nodes.semantic_cache)
    graph.add_node(NODE_TRIAGE_LEAD, nodes.triage_lead)
    graph.add_node(NODE_DB_AGENT, nodes.database_architect_agent)
    graph.add_node(NODE_DEVOPS_AGENT, nodes.devops_agent)
    graph.add_node(NODE_QA_LEAD, nodes.integration_qa_lead)
    graph.add_node(NODE_CACHE_WRITE, nodes.cache_async_write)

    graph.add_edge(START, NODE_SEMANTIC_CACHE)
    graph.add_conditional_edges(
        NODE_SEMANTIC_CACHE,
        _route_after_cache,
        {
            "cache_hit": END,
            "cache_miss": NODE_TRIAGE_LEAD,
        },
    )
    graph.add_conditional_edges(
        NODE_TRIAGE_LEAD,
        _route_after_triage,
        {
            "DB_INFRA": NODE_DB_AGENT,
            "CLOUD_DEVOPS": NODE_DEVOPS_AGENT,
        },
    )
    graph.add_edge(NODE_DB_AGENT, NODE_QA_LEAD)
    graph.add_edge(NODE_DEVOPS_AGENT, NODE_QA_LEAD)
    graph.add_conditional_edges(
        NODE_QA_LEAD,
        lambda state: _route_after_qa(state, runtime_settings.max_qa_retries),
        {
            "qa_passed": NODE_CACHE_WRITE,
            "retry_db": NODE_DB_AGENT,
            "retry_devops": NODE_DEVOPS_AGENT,
            "qa_failed_exhausted": END,
        },
    )
    graph.add_edge(NODE_CACHE_WRITE, END)
    return graph.compile()


def _route_after_cache(state: TriageState) -> str:
    return "cache_hit" if state.get("cache_hit", False) else "cache_miss"


def _route_after_triage(state: TriageState) -> str:
    return state.get("domain_classification", "CLOUD_DEVOPS")


def _route_after_qa(state: TriageState, max_qa_retries: int) -> str:
    if state.get("qa_passed", False):
        return "qa_passed"
    if state.get("retry_count", 0) >= max_qa_retries:
        return "qa_failed_exhausted"
    if state.get("domain_classification") == "DB_INFRA":
        return "retry_db"
    return "retry_devops"

