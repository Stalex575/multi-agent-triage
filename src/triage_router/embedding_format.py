"""Task-aware text formatting for Gemini Embedding 2."""

from __future__ import annotations


def cache_similarity_text(query: str) -> str:
    """Format cache keys consistently for semantic similarity."""

    return f"task: sentence similarity | query: {query.strip()}"


def retrieval_query_text(query: str) -> str:
    """Format a user question for asymmetric context retrieval."""

    return f"task: question answering | query: {query.strip()}"


def retrieval_document_text(title: str, text: str) -> str:
    """Format document text for asymmetric retrieval indexing."""

    safe_title = title.strip() or "none"
    return f"title: {safe_title} | text: {text.strip()}"

