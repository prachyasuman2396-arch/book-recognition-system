"""DecisionAgent -- decides per-detection routing (direct vision vs. super-res).

Thresholds are never hardcoded here; they come from `Settings` (itself
sourced from environment/.env), so ops can retune routing without a
redeploy of agent logic.
"""
from __future__ import annotations

from typing import Any

from app.agents.base_agent import BaseAgent
from app.config import Settings, get_settings
from app.models.domain import QualityDecision, RouteTarget, ToolDecision
from app.models.state import PipelineState


class DecisionAgent(BaseAgent):
    name = "decision_agent"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self.settings = settings or get_settings()

    async def run(self, state: PipelineState) -> dict[str, Any]:
        reports = state.get("quality_reports", [])
        thresholds = {
            "quality_score_accept": self.settings.QUALITY_SCORE_ACCEPT,
            "quality_score_enhance": self.settings.QUALITY_SCORE_ENHANCE,
        }

        decisions: list[ToolDecision] = []
        for report in reports:
            if report.decision == QualityDecision.ACCEPT:
                route = RouteTarget.GEMINI_DIRECT
                reason = (
                    f"quality_score={report.quality_score} >= "
                    f"accept threshold={thresholds['quality_score_accept']}"
                )
            elif report.decision == QualityDecision.ENHANCE:
                route = RouteTarget.SUPER_RESOLUTION
                reason = (
                    f"quality_score={report.quality_score} between enhance "
                    f"threshold={thresholds['quality_score_enhance']} and accept threshold"
                )
            else:
                route = RouteTarget.REJECTED
                reason = f"quality_score={report.quality_score} below enhance threshold"

            decisions.append(
                ToolDecision(
                    detection_id=report.detection_id,
                    route=route,
                    reason=reason,
                    thresholds_used=thresholds,
                )
            )

        return {"tool_decisions": decisions}
