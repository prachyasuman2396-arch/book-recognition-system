from __future__ import annotations

import pytest

from app.agents.decision_agent import DecisionAgent
from app.models.domain import ImageQualityReport, QualityDecision, RouteTarget


def _report(detection_id: str, decision: QualityDecision, score: float) -> ImageQualityReport:
    return ImageQualityReport(
        detection_id=detection_id,
        blur_score=100,
        brightness_score=120,
        contrast_score=40,
        noise_score=5,
        resolution_score=1.0,
        perspective_score=0.9,
        quality_score=score,
        recommendation="test",
        decision=decision,
    )


@pytest.mark.asyncio
async def test_decision_agent_routes_by_quality_decision(tmp_settings):
    agent = DecisionAgent(settings=tmp_settings)
    state = {
        "quality_reports": [
            _report("a", QualityDecision.ACCEPT, 0.9),
            _report("b", QualityDecision.ENHANCE, 0.5),
            _report("c", QualityDecision.REJECT, 0.1),
        ]
    }

    delta = await agent.execute(state)
    decisions = {d.detection_id: d for d in delta["tool_decisions"]}

    assert decisions["a"].route == RouteTarget.GEMINI_DIRECT
    assert decisions["b"].route == RouteTarget.SUPER_RESOLUTION
    assert decisions["c"].route == RouteTarget.REJECTED
    assert delta["execution_trace"][0].agent_name == "decision_agent"
    assert delta["execution_trace"][0].status == "success"
