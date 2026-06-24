"""Configuration loading for the triage router.

All secrets are read from a `.env` file through python-dotenv. Required secrets
are deliberately not read from ambient environment variables so local runs and
deployments are reproducible from explicit configuration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values


DEFAULT_ENV_PATH = Path(".env")


class SettingsError(ValueError):
    """Raised when required runtime settings are missing or invalid."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings for Gemini, Pinecone, and graph behavior."""

    google_api_key: str
    pinecone_api_key: str

    pinecone_index_name: str
    pinecone_cloud: str
    pinecone_region: str
    pinecone_metric: str
    pinecone_cache_namespace: str
    pinecone_db_namespace: str
    pinecone_devops_namespace: str

    heavy_model: str
    fast_model: str
    embedding_model: str
    embedding_dimension: int

    semantic_cache_threshold: float
    max_qa_retries: int
    specialist_top_k: int
    llm_timeout_seconds: int
    llm_max_retries: int
    cache_text_max_chars: int

    @classmethod
    def from_env_file(cls, env_path: str | Path = DEFAULT_ENV_PATH) -> "Settings":
        """Load and validate settings from a dotenv file."""

        path = Path(env_path)
        if not path.exists():
            raise SettingsError(f"Missing .env file at {path.resolve()}")

        raw = dotenv_values(path)

        settings = cls(
            google_api_key=_required(raw, "GOOGLE_API_KEY"),
            pinecone_api_key=_required(raw, "PINECONE_API_KEY"),
            pinecone_index_name=_required(raw, "PINECONE_INDEX_NAME"),
            pinecone_cloud=_get(raw, "PINECONE_CLOUD", "aws"),
            pinecone_region=_get(raw, "PINECONE_REGION", "us-east-1"),
            pinecone_metric=_get(raw, "PINECONE_METRIC", "cosine"),
            pinecone_cache_namespace=_get(raw, "PINECONE_CACHE_NAMESPACE", "semantic-cache"),
            pinecone_db_namespace=_get(raw, "PINECONE_DB_NAMESPACE", "db-infra-context"),
            pinecone_devops_namespace=_get(raw, "PINECONE_DEVOPS_NAMESPACE", "cloud-devops-context"),
            heavy_model=_get(raw, "GEMINI_HEAVY_MODEL", "gemini-3.5-flash"),
            fast_model=_get(raw, "GEMINI_FAST_MODEL", "gemini-3.1-flash-lite"),
            embedding_model=_get(raw, "GEMINI_EMBEDDING_MODEL", "gemini-embedding-2"),
            embedding_dimension=_get_int(raw, "GEMINI_EMBEDDING_DIMENSION", 3072),
            semantic_cache_threshold=_get_float(raw, "SEMANTIC_CACHE_THRESHOLD", 0.92),
            max_qa_retries=_get_int(raw, "MAX_QA_RETRIES", 2),
            specialist_top_k=_get_int(raw, "SPECIALIST_TOP_K", 5),
            llm_timeout_seconds=_get_int(raw, "LLM_TIMEOUT_SECONDS", 60),
            llm_max_retries=_get_int(raw, "LLM_MAX_RETRIES", 2),
            cache_text_max_chars=_get_int(raw, "CACHE_TEXT_MAX_CHARS", 20_000),
        )
        settings.export_for_langchain()
        settings.validate()
        return settings

    def export_for_langchain(self) -> None:
        """Expose only the loaded dotenv values required by LangChain clients."""

        os.environ["GOOGLE_API_KEY"] = self.google_api_key

    def validate(self) -> None:
        """Validate constraints that can fail silently in downstream services."""

        if not 0.0 <= self.semantic_cache_threshold <= 1.0:
            raise SettingsError("SEMANTIC_CACHE_THRESHOLD must be between 0 and 1")
        if self.max_qa_retries < 0:
            raise SettingsError("MAX_QA_RETRIES must be >= 0")
        if self.specialist_top_k < 1:
            raise SettingsError("SPECIALIST_TOP_K must be >= 1")
        if self.embedding_dimension < 1:
            raise SettingsError("GEMINI_EMBEDDING_DIMENSION must be positive")
        if self.pinecone_metric != "cosine":
            raise SettingsError("PINECONE_METRIC must be cosine for the configured similarity threshold")


def _required(raw: Mapping[str, str | None], key: str) -> str:
    value = raw.get(key)
    if value is None or value.strip() == "":
        raise SettingsError(f"Missing required .env variable: {key}")
    return value.strip()


def _get(raw: Mapping[str, str | None], key: str, default: str) -> str:
    value = raw.get(key)
    return default if value is None or value.strip() == "" else value.strip()


def _get_int(raw: Mapping[str, str | None], key: str, default: int) -> int:
    value = _get(raw, key, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise SettingsError(f"{key} must be an integer") from exc


def _get_float(raw: Mapping[str, str | None], key: str, default: float) -> float:
    value = _get(raw, key, str(default))
    try:
        return float(value)
    except ValueError as exc:
        raise SettingsError(f"{key} must be a float") from exc

