"""Seed Pinecone with small DB and DevOps context corpora."""

from __future__ import annotations

import asyncio

from triage_router.config import Settings
from triage_router.embedding_format import retrieval_document_text
from triage_router.models import build_models
from triage_router.pinecone_store import PineconeStore


DB_DOCUMENTS = [
    {
        "title": "Postgres write latency triage",
        "source": "seed://db/postgres-write-latency",
        "text": (
            "For PostgreSQL write latency spikes, inspect pg_stat_activity, pg_locks, "
            "pg_stat_bgwriter, WAL generation, checkpoint frequency, autovacuum lag, "
            "index bloat, long transactions, and connection pool saturation. Validate "
            "with EXPLAIN (ANALYZE, BUFFERS), pg_stat_statements, and p95 commit latency."
        ),
    },
    {
        "title": "MySQL online index rollout safety",
        "source": "seed://db/mysql-online-index",
        "text": (
            "For MySQL index changes, prefer online DDL or gh-ost/pt-online-schema-change, "
            "monitor metadata locks, replica lag, buffer pool pressure, and query plan flips. "
            "Keep rollback DDL, throttle copy rate, and validate with EXPLAIN FORMAT=JSON."
        ),
    },
    {
        "title": "Database connection pool sizing",
        "source": "seed://db/pool-sizing",
        "text": (
            "Set application pool sizes from database worker capacity, not pod count alone. "
            "Use PgBouncer transaction pooling where appropriate, cap max connections, "
            "alert on wait time, idle-in-transaction sessions, and saturation."
        ),
    },
]


DEVOPS_DOCUMENTS = [
    {
        "title": "Kubernetes blue-green deployment checklist",
        "source": "seed://devops/k8s-blue-green",
        "text": (
            "A Kubernetes blue-green rollout should isolate deployments with labels, switch "
            "service selectors atomically, run readiness gates, validate smoke tests, preserve "
            "the old deployment for rollback, and monitor error rate, latency, saturation, and "
            "business KPIs during the traffic switch."
        ),
    },
    {
        "title": "Terraform production safety controls",
        "source": "seed://devops/terraform-safety",
        "text": (
            "Production Terraform pipelines should use remote state locking, plan artifacts, "
            "policy checks, drift detection, manual approval for sensitive workspaces, least "
            "privilege cloud credentials, and staged applies with rollback documentation."
        ),
    },
    {
        "title": "CI/CD container promotion",
        "source": "seed://devops/cicd-promotion",
        "text": (
            "Container delivery should build once, scan SBOM and vulnerabilities, sign images, "
            "promote immutable digests across environments, and deploy with health checks, "
            "progressive traffic shifting, and automated rollback conditions."
        ),
    },
]


async def main() -> None:
    """Embed and upsert sample retrieval context documents."""

    settings = Settings.from_env_file()
    models = build_models(settings)
    store = PineconeStore(settings, models.embeddings)

    await store.upsert_context_documents(
        namespace=settings.pinecone_db_namespace,
        documents=_prepare(DB_DOCUMENTS),
    )
    await store.upsert_context_documents(
        namespace=settings.pinecone_devops_namespace,
        documents=_prepare(DEVOPS_DOCUMENTS),
    )
    print("Seeded DB_INFRA and CLOUD_DEVOPS context namespaces.")


def _prepare(documents: list[dict[str, str]]) -> list[dict[str, str]]:
    prepared = []
    for document in documents:
        item = dict(document)
        item["embedding_text"] = retrieval_document_text(item["title"], item["text"])
        prepared.append(item)
    return prepared


if __name__ == "__main__":
    asyncio.run(main())

