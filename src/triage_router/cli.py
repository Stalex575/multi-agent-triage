"""Command-line entry point."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from triage_router.config import Settings
from triage_router.graph import build_app
from triage_router.state import make_initial_state


async def _run(query: str, env_file: Path) -> None:
    settings = Settings.from_env_file(env_file)
    app = build_app(settings)
    result = await app.ainvoke(make_initial_state(query))
    print(result.get("final_response", ""))


def main() -> None:
    """Run the triage router for a single query."""

    parser = argparse.ArgumentParser(description="Run the multi-agent triage router.")
    parser.add_argument("query", help="Infrastructure question to route and answer.")
    parser.add_argument("--env-file", default=".env", help="Path to the dotenv file.")
    args = parser.parse_args()

    asyncio.run(_run(args.query, Path(args.env_file)))


if __name__ == "__main__":
    main()

