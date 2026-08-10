"""GeminiVisionTool -- batched, schema-constrained book recognition.

Uses Google's official `google-genai` SDK. Multiple crops are sent in a
single request (up to `GEMINI_BATCH_SIZE`) with a structured-output schema
so Gemini returns strict JSON, never free-form prose.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings, get_settings
from app.core.exceptions import GeminiAPIError
from app.models.domain import TokenUsage, VisionExtraction
from app.tools.base_tool import BaseTool

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "books": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "detection_id": {"type": "string"},
                    "title": {"type": "string"},
                    "author": {"type": "string"},
                    "confidence": {"type": "number"},
                    "reasoning": {"type": "string"},
                    "language": {"type": "string"},
                    "visible_text": {"type": "string"},
                },
                "required": [
                    "detection_id", "title", "author", "confidence",
                    "reasoning", "language", "visible_text",
                ],
            },
        }
    },
    "required": ["books"],
}

_SYSTEM_PROMPT = (
    "You are a precise book-cover and book-spine recognition system. "
    "For each labeled image, identify the exact book title and author as "
    "printed on the cover/spine. If text is partially occluded, use only "
    "what is visible and lower your confidence accordingly. Never invent "
    "a title or author you cannot justify from visible text. Respond only "
    "with JSON matching the provided schema -- no prose, no markdown fences."
)


@dataclass
class GeminiVisionInput:
    image_paths: dict[str, str]  # detection_id -> image path (crop or enhanced)


@dataclass
class GeminiVisionOutput:
    extractions: list[VisionExtraction]
    token_usage: TokenUsage


class GeminiVisionTool(BaseTool[GeminiVisionInput, list[GeminiVisionOutput]]):
    name = "gemini_vision_tool"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        self.timeout_seconds = self.settings.GEMINI_TIMEOUT_SECONDS
        self.max_retries = self.settings.GEMINI_MAX_RETRIES
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        from google import genai  # official Google GenAI SDK

        self._client = genai.Client(api_key=self.settings.GEMINI_API_KEY)
        return self._client

    async def run(self, payload: GeminiVisionInput) -> list[GeminiVisionOutput]:
        return await self._execute(self._recognize_all, payload, retry_on=(GeminiAPIError,))

    async def _recognize_all(self, payload: GeminiVisionInput) -> list[GeminiVisionOutput]:
        items = list(payload.image_paths.items())
        batch_size = self.settings.GEMINI_BATCH_SIZE
        batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

        outputs: list[GeminiVisionOutput] = []
        for batch in batches:
            outputs.append(await self._recognize_batch(batch))
        return outputs

    async def _recognize_batch(
        self, batch: list[tuple[str, str]]
    ) -> GeminiVisionOutput:
        import asyncio

        client = self._get_client()
        contents = self._build_contents(batch)

        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=self.settings.GEMINI_MODEL,
                    contents=contents,
                    config={
                        "system_instruction": _SYSTEM_PROMPT,
                        "temperature": self.settings.GEMINI_TEMPERATURE,
                        "max_output_tokens": self.settings.GEMINI_MAX_OUTPUT_TOKENS,
                        "response_mime_type": "application/json",
                        "response_schema": _RESPONSE_SCHEMA,
                    },
                ),
            )
        except Exception as exc:  # noqa: BLE001
            raise GeminiAPIError(f"Gemini request failed: {exc}") from exc

        parsed = self._parse_response(response)
        usage = self._extract_token_usage(response)
        return GeminiVisionOutput(extractions=parsed, token_usage=usage)

    def _build_contents(self, batch: list[tuple[str, str]]) -> list[dict]:
        parts: list[dict] = [
            {
                "text": (
                    "Analyze the following labeled book images. Return one entry per "
                    "detection_id in the `books` array of your JSON response. "
                    "detection_ids: " + ", ".join(det_id for det_id, _ in batch)
                )
            }
        ]
        for detection_id, path in batch:
            image_bytes = Path(path).read_bytes()
            parts.append({"text": f"[detection_id={detection_id}]"})
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_bytes}})
        return [{"role": "user", "parts": parts}]

    def _parse_response(self, response) -> list[VisionExtraction]:
        try:
            raw_text = response.text
            data = json.loads(raw_text)
        except (AttributeError, json.JSONDecodeError, TypeError) as exc:
            raise GeminiAPIError(f"Could not parse Gemini structured response: {exc}") from exc

        extractions: list[VisionExtraction] = []
        for entry in data.get("books", []):
            try:
                extractions.append(VisionExtraction(**entry))
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Skipping malformed Gemini entry: %s", exc)
        return extractions

    def _extract_token_usage(self, response) -> TokenUsage:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return TokenUsage(model=self.settings.GEMINI_MODEL, input_tokens=0, output_tokens=0)
        return TokenUsage(
            model=self.settings.GEMINI_MODEL,
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        )
