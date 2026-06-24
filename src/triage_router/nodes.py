"""LangGraph node implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from triage_router.config import Settings
from triage_router.models import ModelBundle
from triage_router.pinecone_store import PineconeStore
from triage_router.prompts import (
    DB_SPECIALIST_SYSTEM_PROMPT,
    DEVOPS_SPECIALIST_SYSTEM_PROMPT,
    QA_SYSTEM_PROMPT,
    TRIAGE_SYSTEM_PROMPT,
    qa_user_prompt,
    specialist_user_prompt,
)
from triage_router.state import DomainClassification, TriageState


class DomainClassificationResult(BaseModel):
    """Structured result returned by the Triage Lead."""

    domain_classification: Literal["DB_INFRA", "CLOUD_DEVOPS"] = Field(
        description="Exactly one domain classification."
    )


class QAResult(BaseModel):
    """Structured result returned by the Integration QA Lead."""

    qa_passed: bool = Field(description="Whether the draft meets the production rubric.")
    feedback: str = Field(description="Actionable feedback when qa_passed is false.")
    final_response: str = Field(description="Polished final answer when qa_passed is true.")


@dataclass(slots=True)
class TriageNodes:
    """Callable graph nodes with shared service dependencies."""

    settings: Settings
    models: ModelBundle
    store: PineconeStore

    async def semantic_cache(self, state: TriageState) -> dict[str, object]:
        """Gatekeeper node that returns a semantic cache hit when available."""

        lookup = await self.store.lookup_cache(state["query"])
        if lookup.hit:
            return {
                "cache_hit": True,
                "final_response": lookup.final_response,
                "qa_passed": True,
            }
        return {"cache_hit": False}

    async def triage_lead(self, state: TriageState) -> dict[str, object]:
        """Classify the query into the specialist domain."""

        classifier = self.models.router.with_structured_output(DomainClassificationResult)
        result = await classifier.ainvoke(
            [
                ("system", TRIAGE_SYSTEM_PROMPT),
                ("human", state["query"]),
            ]
        )
        return {"domain_classification": result.domain_classification}

    async def database_architect_agent(self, state: TriageState) -> dict[str, object]:
        """Retrieve DB context and draft a database infrastructure solution."""

        return await self._specialist_node(
            state=state,
            domain="DB_INFRA",
            system_prompt=DB_SPECIALIST_SYSTEM_PROMPT,
        )

    async def devops_agent(self, state: TriageState) -> dict[str, object]:
        """Retrieve DevOps context and draft a cloud infrastructure solution."""

        return await self._specialist_node(
            state=state,
            domain="CLOUD_DEVOPS",
            system_prompt=DEVOPS_SPECIALIST_SYSTEM_PROMPT,
        )

    async def integration_qa_lead(self, state: TriageState) -> dict[str, object]:
        """Validate the specialist draft and produce retry feedback if needed."""

        draft = state.get("draft_solution", "")
        qa = self.models.qa.with_structured_output(QAResult)
        result = await qa.ainvoke(
            [
                ("system", QA_SYSTEM_PROMPT),
                ("human", qa_user_prompt(query=state["query"], draft_solution=draft)),
            ]
        )

        if result.qa_passed:
            return {
                "qa_passed": True,
                "qa_feedback": "",
                "final_response": result.final_response.strip() or draft,
            }

        exhausted = state.get("retry_count", 0) >= self.settings.max_qa_retries
        final_response = ""
        if exhausted:
            final_response = (
                "QA did not pass after the configured retry budget. Latest draft follows.\n\n"
                f"{draft}\n\nQA feedback:\n{result.feedback.strip()}"
            )

        return {
            "qa_passed": False,
            "qa_feedback": result.feedback.strip(),
            "final_response": final_response,
        }

    async def cache_async_write(self, state: TriageState) -> dict[str, object]:
        """Persist a successful novel response to Pinecone cache."""

        domain = state.get("domain_classification", "CLOUD_DEVOPS")
        final_response = state.get("final_response", "")
        if final_response and state.get("qa_passed", False):
            await self.store.write_cache(
                query=state["query"],
                domain=domain,
                final_response=final_response,
            )
        return {}

    async def _specialist_node(
        self,
        *,
        state: TriageState,
        domain: DomainClassification,
        system_prompt: str,
    ) -> dict[str, object]:
        previous_feedback = state.get("qa_feedback", "")
        retry_count = state.get("retry_count", 0) + (1 if previous_feedback else 0)
        context = await self.store.retrieve_context(state["query"], domain)
        response = await self.models.specialist.ainvoke(
            [
                ("system", system_prompt),
                (
                    "human",
                    specialist_user_prompt(
                        query=state["query"],
                        retrieved_context=context,
                        qa_feedback=previous_feedback,
                        retry_count=retry_count,
                    ),
                ),
            ]
        )
        return {
            "domain_classification": domain,
            "retrieved_context": context,
            "draft_solution": _message_text(response),
            "retry_count": retry_count,
            "qa_passed": False,
        }


def _message_text(response: object) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    return str(content).strip()

