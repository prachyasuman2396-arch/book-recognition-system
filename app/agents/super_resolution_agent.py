"""SuperResolutionAgent -- enhances only the crops routed to it by DecisionAgent."""
from __future__ import annotations

import asyncio
from typing import Any

from app.agents.base_agent import BaseAgent
from app.models.domain import RouteTarget
from app.models.state import PipelineState
from app.tools.super_resolution_tool import SuperResolutionInput, SuperResolutionTool


class SuperResolutionAgent(BaseAgent):
    name = "super_resolution_agent"

    def __init__(self, super_res_tool: SuperResolutionTool) -> None:
        super().__init__()
        self.super_res_tool = super_res_tool

    async def run(self, state: PipelineState) -> dict[str, Any]:
        decisions = state.get("tool_decisions", [])
        detections = {d.detection_id: d for d in state.get("detections", [])}
        request_id = state.get("request_id", "unknown")

        targets = [d for d in decisions if d.route == RouteTarget.SUPER_RESOLUTION]
        if not targets:
            return {"enhanced_images": []}

        enhanced = await asyncio.gather(
            *(
                self.super_res_tool.run(
                    SuperResolutionInput(
                        detection_id=decision.detection_id,
                        crop_path=detections[decision.detection_id].crop_path,
                        request_id=request_id,
                    )
                )
                for decision in targets
                if decision.detection_id in detections
            )
        )
        return {"enhanced_images": list(enhanced)}
