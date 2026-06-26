"""FastAPI web service for the Multi-Agent Triage Router.

Exposes the LangGraph triage workflow over HTTP so it can be deployed
on AWS ECS Fargate (or any container runtime) behind port 8000.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from triage_router.config import Settings
from triage_router.graph import build_app
from triage_router.state import make_initial_state

logger = logging.getLogger("triage_router.api")


class TriageRequest(BaseModel):
    """Incoming triage query."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The infrastructure question to route and answer.",
    )


class TriageResponse(BaseModel):
    """Outgoing triage response."""

    query: str = Field(description="The original user query.")
    response: str = Field(description="The final triage response.")
    elapsed_seconds: float = Field(description="Wall-clock time in seconds.")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown lifecycle hook.

    Initialises ``Settings`` and compiles the LangGraph application once,
    storing them on ``app.state`` so request handlers can access them
    without module-level globals.
    """

    logger.info("Starting up — loading settings and compiling graph…")
    settings = Settings.load()
    triage_app = build_app(settings)
    app.state.settings = settings
    app.state.triage_app = triage_app
    logger.info("Graph compiled — ready to serve requests.")

    yield

    logger.info("Shutting down.")


app = FastAPI(
    title="Multi-Agent Triage Router API",
    description="Production API for the LangGraph multi-agent triage system.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Liveness / readiness probe for ECS Fargate and load balancers."""

    return HealthResponse(status="healthy")


@app.post("/api/triage", response_model=TriageResponse)
async def run_triage(body: TriageRequest, request: Request) -> TriageResponse:
    """Invoke the LangGraph triage workflow for a user query.

    The compiled graph is retrieved from ``app.state`` (initialised once
    during the lifespan startup phase).
    """

    triage_app = request.app.state.triage_app

    logger.info("Received triage request: %s", body.query[:120])
    start = time.perf_counter()

    try:
        initial_state = make_initial_state(body.query)
        result = await triage_app.ainvoke(initial_state)
    except ValueError as exc:
        # make_initial_state raises ValueError on empty / blank queries
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Graph invocation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = time.perf_counter() - start
    final_response = result.get("final_response", "")
    logger.info("Triage completed in %.2fs", elapsed)

    return TriageResponse(
        query=body.query,
        response=final_response,
        elapsed_seconds=round(elapsed, 3),
    )
