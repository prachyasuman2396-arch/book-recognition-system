"""GeminiVisionAgent -- structured book recognition via Gemini Vision.

Picks, per detection, the *best available* image: the super-resolution
enhanced version if one exists, otherwise the original crop. Rejected
detections (RouteTarget.REJECTED) are excluded entirely.
"""
from __future__ import annotations

from typing import Any

from app.agents.base_agent import BaseAgent
from app.config import Settings, get_settings
from app.models.domain import RouteTarget
from app.models.state import PipelineState
from app.tools.gemini_vision_tool import GeminiVisionInput, GeminiVisionTool


class GeminiVisionAgent(BaseAgent):
    name = "gemini_vision_agent"

    def __init__(self, gemini_tool: GeminiVisionTool, settings: Settings | None = None) -> None:
        super().__init__()
        self.gemini_tool = gemini_tool
        self.settings = settings or get_settings()

    async def run(self, state: PipelineState) -> dict[str, Any]:
        decisions = {d.detection_id: d for d in state.get("tool_decisions", [])}
        detections = {d.detection_id: d for d in state.get("detections", [])}
        enhanced = {e.detection_id: e for e in state.get("enhanced_images", [])}

        image_paths: dict[str, str] = {}
        for detection_id, decision in decisions.items():
            if decision.route == RouteTarget.REJECTED:
                continue
            if detection_id in enhanced:
                image_paths[detection_id] = enhanced[detection_id].enhanced_path
            elif detection_id in detections:
                image_paths[detection_id] = detections[detection_id].crop_path

        if not image_paths:
            return {"vision_results": [], "warnings": ["No eligible images for Vision LLM analysis"]}

        batches = await self.gemini_tool.run(GeminiVisionInput(image_paths=image_paths))

        vision_results = []
        token_usage = []
        estimated_cost = 0.0
        for batch in batches:
            vision_results.extend(batch.extractions)
            token_usage.append(batch.token_usage)
            estimated_cost += self._estimate_cost(batch.token_usage)

        return {
            "vision_results": vision_results,
            "token_usage": token_usage,
            "estimated_cost_usd": round(estimated_cost, 6),
        }

    def _estimate_cost(self, usage) -> float:
        input_cost = (usage.input_tokens / 1000) * self.settings.GEMINI_COST_PER_1K_INPUT_TOKENS_USD
        output_cost = (usage.output_tokens / 1000) * self.settings.GEMINI_COST_PER_1K_OUTPUT_TOKENS_USD
        return input_cost + output_cost
