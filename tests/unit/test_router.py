from __future__ import annotations

from app.graph.router import (
    route_after_decision,
    route_after_detection,
    route_after_validation,
)
from app.models.domain import RouteTarget, ToolDecision


def _decision(route: RouteTarget) -> ToolDecision:
    return ToolDecision(detection_id="x", route=route, reason="test", thresholds_used={})


def test_route_after_detection_no_detections():
    assert route_after_detection({"detections": []}) == "no_detections"


def test_route_after_detection_has_detections():
    assert route_after_detection({"detections": [object()]}) == "has_detections"


def test_route_after_decision_all_rejected():
    state = {"tool_decisions": [_decision(RouteTarget.REJECTED), _decision(RouteTarget.REJECTED)]}
    assert route_after_decision(state) == "all_rejected"


def test_route_after_decision_needs_super_resolution():
    state = {
        "tool_decisions": [
            _decision(RouteTarget.GEMINI_DIRECT),
            _decision(RouteTarget.SUPER_RESOLUTION),
        ]
    }
    assert route_after_decision(state) == "needs_super_resolution"


def test_route_after_decision_direct_to_vision():
    state = {"tool_decisions": [_decision(RouteTarget.GEMINI_DIRECT)]}
    assert route_after_decision(state) == "direct_to_vision"


def test_route_after_validation_none_validated():
    assert route_after_validation({"validated_books": []}) == "no_validated_books"


def test_route_after_validation_has_validated():
    assert route_after_validation({"validated_books": [object()]}) == "has_validated_books"
