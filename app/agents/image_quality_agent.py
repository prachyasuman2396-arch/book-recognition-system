"""ImageQualityAgent -- scores every detected crop for downstream routing."""
from __future__ import annotations

import asyncio
from typing import Any

from app.agents.base_agent import BaseAgent
from app.models.state import PipelineState
from app.tools.image_quality_tool import ImageQualityInput, ImageQualityTool


class ImageQualityAgent(BaseAgent):
    name = "image_quality_agent"

    def __init__(self, quality_tool: ImageQualityTool) -> None:
        super().__init__()
        self.quality_tool = quality_tool

    async def run(self, state: PipelineState) -> dict[str, Any]:
        detections = state.get("detections", [])
        if not detections:
            return {"quality_reports": []}

        reports = await asyncio.gather(
            *(
                self.quality_tool.run(
                    ImageQualityInput(detection_id=d.detection_id, crop_path=d.crop_path)
                )
                for d in detections
            )
        )
        return {"quality_reports": list(reports)}
