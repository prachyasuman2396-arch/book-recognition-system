"""Conditional-edge routing functions for the pipeline `StateGraph`.

Each function inspects the current merged `PipelineState` and returns one
of a small set of string labels used in `add_conditional_edges` path maps.
Kept separate from `workflow.py` so routing logic is independently
unit-testable without constructing a graph.
"""
from __future__ import annotations

from app.models.domain import RouteTarget
from app.models.state import PipelineState


def route_after_detection(state: PipelineState) -> str:
    """Skip straight to the end if nothing was detected."""
    if not state.get("detections"):
        return "no_detections"
    return "has_detections"


def route_after_decision(state: PipelineState) -> str:
    """Decide whether any crop needs super-resolution before Gemini.

    If *every* decision is GEMINI_DIRECT or REJECTED, skip the
    SuperResolutionAgent node entirely to save latency/cost.
    """
    decisions = state.get("tool_decisions", [])
    needs_super_res = any(d.route == RouteTarget.SUPER_RESOLUTION for d in decisions)
    all_rejected = decisions and all(d.route == RouteTarget.REJECTED for d in decisions)

    if all_rejected:
        return "all_rejected"
    if needs_super_res:
        return "needs_super_resolution"
    return "direct_to_vision"


def route_after_validation(state: PipelineState) -> str:
    """Skip recommendations if nothing survived validation."""
    if not state.get("validated_books"):
        return "no_validated_books"
    return "has_validated_books"
