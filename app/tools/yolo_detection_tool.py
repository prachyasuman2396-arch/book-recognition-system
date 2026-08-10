"""YOLODetectionTool -- detects book spines/covers in an uploaded image.

Wraps Ultralytics YOLO. Falls back to a deterministic single-box detector
when the `ultralytics` package or model weights are unavailable, so the
rest of the pipeline (and tests) can run without a GPU or downloaded
weights. The fallback is explicit and logged -- never silent.
"""
from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from app.config import Settings, get_settings
from app.core.exceptions import ToolError
from app.models.domain import BoundingBox, Detection
from app.tools.base_tool import BaseTool


@dataclass
class YoloDetectionInput:
    image_path: str
    request_id: str


class YOLODetectionTool(BaseTool[YoloDetectionInput, list[Detection]]):
    """Stateless tool: image path in, list of `Detection` out."""

    name = "yolo_detection_tool"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        self.timeout_seconds = self.settings.TIMEOUT
        self.max_retries = self.settings.MAX_RETRIES
        self._model = None  # lazily loaded, shared across calls (read-only)

    def _load_model(self):  # -> ultralytics.YOLO | None
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO  # type: ignore

            model_path = Path(self.settings.YOLO_MODEL_PATH)
            if not model_path.exists():
                self.logger.warning(
                    "YOLO weights not found at %s; using heuristic fallback detector",
                    model_path,
                )
                return None
            self._model = YOLO(str(model_path))
            return self._model
        except ImportError:
            self.logger.warning("ultralytics not installed; using heuristic fallback detector")
            return None

    def warm_up(self) -> None:
        """Synchronously load weights. Call from a worker thread only
        (see `Container.warm_up`) -- never from the event loop directly."""
        self._load_model()

    async def run(self, payload: YoloDetectionInput) -> list[Detection]:
        return await self._execute(self._detect, payload)

    async def _detect(self, payload: YoloDetectionInput) -> list[Detection]:
        # `_detect_sync` does blocking, CPU-bound work: weight loading (on
        # first call), YOLO inference, and synchronous disk I/O for crops.
        # Running that directly inside this `async def` with no `await`
        # would block the entire event loop -- every other in-flight
        # request on this worker -- for the full duration. `asyncio.to_thread`
        # runs it in a worker thread instead, keeping the loop free.
        return await asyncio.to_thread(self._detect_sync, payload)

    def _detect_sync(self, payload: YoloDetectionInput) -> list[Detection]:
        image_path = Path(payload.image_path)
        if not image_path.exists():
            raise ToolError(f"Image not found: {image_path}")

        model = self._load_model()
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            width, height = img.size

            if model is not None:
                boxes = self._run_yolo_inference(model, image_path)
            else:
                boxes = self._fallback_full_image_box(width, height)

            detections: list[Detection] = []
            crop_dir = Path(self.settings.CROP_DIR) / payload.request_id
            crop_dir.mkdir(parents=True, exist_ok=True)

            for box in boxes[: self.settings.YOLO_MAX_DETECTIONS]:
                padded_box = self._apply_padding(box, width, height)
                detection_id = str(uuid.uuid4())[:8]
                crop = img.crop(
                    (
                        int(padded_box.x_min),
                        int(padded_box.y_min),
                        int(padded_box.x_max),
                        int(padded_box.y_max),
                    )
                )
                crop_path = crop_dir / f"{detection_id}.jpg"
                crop.save(crop_path, format="JPEG", quality=95)

                detections.append(
                    Detection(
                        detection_id=detection_id,
                        bbox=padded_box,
                        confidence=box[4],
                        crop_path=str(crop_path),
                        padded=True,
                    )
                )
        return detections

    def _run_yolo_inference(self, model, image_path: Path) -> list[tuple[float, float, float, float, float]]:
        results = model.predict(
            source=str(image_path),
            conf=self.settings.YOLO_CONFIDENCE_THRESHOLD,
            iou=self.settings.YOLO_IOU_THRESHOLD,
            device=self.settings.YOLO_DEVICE,
            verbose=False,
        )
        boxes: list[tuple[float, float, float, float, float]] = []
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                boxes.append((xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf))
        boxes.sort(key=lambda b: b[4], reverse=True)
        return boxes

    def _fallback_full_image_box(
        self, width: int, height: int
    ) -> list[tuple[float, float, float, float, float]]:
        """Deterministic fallback: treat the whole image as one candidate."""
        return [(0.0, 0.0, float(width), float(height), 0.5)]

    def _apply_padding(
        self, box: tuple[float, float, float, float, float], width: int, height: int
    ) -> BoundingBox:
        x_min, y_min, x_max, y_max, _ = box
        pad_x = (x_max - x_min) * self.settings.YOLO_CROP_PADDING_RATIO
        pad_y = (y_max - y_min) * self.settings.YOLO_CROP_PADDING_RATIO
        return BoundingBox(
            x_min=max(0.0, x_min - pad_x),
            y_min=max(0.0, y_min - pad_y),
            x_max=min(float(width), x_max + pad_x),
            y_max=min(float(height), y_max + pad_y),
        )
