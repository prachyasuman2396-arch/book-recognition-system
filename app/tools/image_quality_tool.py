"""ImageQualityTool -- scores a cropped book image on several axes.

Pure OpenCV/NumPy, no ML model required, so it's fast and deterministic
(important since `DecisionAgent` routes on these scores).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.config import Settings, get_settings
from app.core.exceptions import ToolError
from app.models.domain import ImageQualityReport, QualityDecision
from app.tools.base_tool import BaseTool


@dataclass
class ImageQualityInput:
    detection_id: str
    crop_path: str


class ImageQualityTool(BaseTool[ImageQualityInput, ImageQualityReport]):
    name = "image_quality_tool"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        self.timeout_seconds = 10.0
        self.max_retries = 1  # deterministic local computation; retries add no value

    async def run(self, payload: ImageQualityInput) -> ImageQualityReport:
        return await self._execute(self._assess, payload)

    async def _assess(self, payload: ImageQualityInput) -> ImageQualityReport:
        image = cv2.imread(payload.crop_path)
        if image is None:
            raise ToolError(f"Could not read image for quality check: {payload.crop_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]

        blur_score = self._laplacian_variance(gray)
        brightness_score = float(np.mean(gray))
        contrast_score = float(np.std(gray))
        noise_score = self._estimate_noise(gray)
        resolution_score = self._resolution_score(width, height)
        perspective_score = self._perspective_score(gray)

        quality_score = self._composite_score(
            blur_score, brightness_score, contrast_score, noise_score,
            resolution_score, perspective_score,
        )
        decision, recommendation = self._decide(quality_score)

        return ImageQualityReport(
            detection_id=payload.detection_id,
            blur_score=round(blur_score, 2),
            brightness_score=round(brightness_score, 2),
            contrast_score=round(contrast_score, 2),
            noise_score=round(noise_score, 2),
            resolution_score=round(resolution_score, 2),
            perspective_score=round(perspective_score, 2),
            quality_score=round(quality_score, 4),
            recommendation=recommendation,
            decision=decision,
        )

    @staticmethod
    def _laplacian_variance(gray: np.ndarray) -> float:
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _estimate_noise(gray: np.ndarray) -> float:
        median = cv2.medianBlur(gray, 3)
        diff = cv2.absdiff(gray, median)
        return float(np.mean(diff))

    def _resolution_score(self, width: int, height: int) -> float:
        min_w, min_h = self.settings.MIN_WIDTH, self.settings.MIN_HEIGHT
        return min(1.0, (width * height) / max(1, (min_w * min_h)))

    @staticmethod
    def _perspective_score(gray: np.ndarray) -> float:
        """Approximate how "rectangular"/frontal the crop is via edge angles."""
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=40, maxLineGap=10)
        if lines is None or len(lines) == 0:
            return 0.5  # inconclusive, treat as neutral
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(abs(angle) % 90)
        # angles close to 0 or 90 => axis-aligned => good perspective
        deviations = [min(a, 90 - a) for a in angles]
        avg_deviation = float(np.mean(deviations)) if deviations else 45.0
        return max(0.0, 1.0 - (avg_deviation / 45.0))

    def _composite_score(
        self,
        blur: float,
        brightness: float,
        contrast: float,
        noise: float,
        resolution: float,
        perspective: float,
    ) -> float:
        s = self.settings
        blur_norm = min(1.0, blur / s.BLUR_THRESHOLD)
        brightness_norm = 1.0 if s.BRIGHTNESS_MIN <= brightness <= s.BRIGHTNESS_MAX else 0.4
        contrast_norm = min(1.0, contrast / s.CONTRAST_THRESHOLD)
        noise_norm = max(0.0, 1.0 - (noise / max(s.NOISE_THRESHOLD, 1e-6)))
        # weighted blend; weights sum to 1.0
        return (
            0.30 * blur_norm
            + 0.15 * brightness_norm
            + 0.15 * contrast_norm
            + 0.15 * noise_norm
            + 0.15 * resolution
            + 0.10 * perspective
        )

    def _decide(self, quality_score: float) -> tuple[QualityDecision, str]:
        s = self.settings
        if quality_score >= s.QUALITY_SCORE_ACCEPT:
            return QualityDecision.ACCEPT, "Quality sufficient for direct Vision LLM analysis"
        if quality_score >= s.QUALITY_SCORE_ENHANCE:
            return QualityDecision.ENHANCE, "Quality marginal; recommend super-resolution before Vision LLM"
        return QualityDecision.REJECT, "Quality too poor for reliable recognition even after enhancement"
