"""BookDetectionAgent -- detects, pads, and crops book regions."""
from __future__ import annotations

from typing import Any

from app.agents.base_agent import BaseAgent
from app.core.exceptions import AgentExecutionError
from app.models.state import PipelineState
from app.tools.yolo_detection_tool import YOLODetectionTool, YoloDetectionInput


class BookDetectionAgent(BaseAgent):
    name = "book_detection_agent"

    def __init__(self, yolo_tool: YOLODetectionTool) -> None:
        super().__init__()
        self.yolo_tool = yolo_tool

    async def run(self, state: PipelineState) -> dict[str, Any]:
        image_path = state.get("original_image_path")
        request_id = state.get("request_id", "unknown")
        if not image_path:
            raise AgentExecutionError("No original_image_path present in state")

        detections = await self.yolo_tool.run(
            YoloDetectionInput(image_path=image_path, request_id=request_id)
        )

        if not detections:
            return {
                "detections": [],
                "cropped_images": [],
                "warnings": ["No books detected in the uploaded image"],
            }

        return {
            "detections": detections,
            "cropped_images": [d.crop_path for d in detections],
        }
