#!/usr/bin/env python3
"""Regenerate `docs/architecture-graph.mmd` from the live LangGraph definition.

Run with: python scripts/generate_graph_diagram.py
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph.runner import render_graph_mermaid  # noqa: E402


async def main() -> None:
    mermaid = await render_graph_mermaid()
    out_path = Path(__file__).resolve().parent.parent / "docs" / "architecture-graph.mmd"
    out_path.write_text(mermaid)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
