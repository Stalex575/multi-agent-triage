"""Gemini model factories."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from triage_router.config import Settings


@dataclass(frozen=True, slots=True)
class ModelBundle:
    """Initialized LangChain model clients used by graph nodes."""

    router: ChatGoogleGenerativeAI
    specialist: ChatGoogleGenerativeAI
    qa: ChatGoogleGenerativeAI
    embeddings: GoogleGenerativeAIEmbeddings


def build_models(settings: Settings) -> ModelBundle:
    """Build Gemini clients with model names pinned by configuration."""

    router = ChatGoogleGenerativeAI(
        model=settings.fast_model,
        temperature=0.0,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
    specialist = ChatGoogleGenerativeAI(
        model=settings.heavy_model,
        temperature=0.2,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
    qa = ChatGoogleGenerativeAI(
        model=settings.fast_model,
        temperature=0.0,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        output_dimensionality=settings.embedding_dimension,
    )

    return ModelBundle(router=router, specialist=specialist, qa=qa, embeddings=embeddings)

