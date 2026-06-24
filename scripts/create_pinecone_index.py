"""Create the Pinecone dense vector index required by the router."""

from __future__ import annotations

import time

from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone

from triage_router.config import Settings


def main() -> None:
    """Create the configured Pinecone index if it does not exist."""

    settings = Settings.from_env_file()
    pc = Pinecone(api_key=settings.pinecone_api_key)

    if not pc.has_index(settings.pinecone_index_name):
        pc.create_index(
            name=settings.pinecone_index_name,
            vector_type="dense",
            dimension=settings.embedding_dimension,
            metric=settings.pinecone_metric,
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
            deletion_protection="disabled",
            tags={"application": "multi-agent-triage-router"},
        )

    while not pc.describe_index(settings.pinecone_index_name).status["ready"]:
        time.sleep(2)

    print(
        "Pinecone index ready: "
        f"{settings.pinecone_index_name} "
        f"dimension={settings.embedding_dimension} "
        f"metric={settings.pinecone_metric}"
    )


if __name__ == "__main__":
    main()

