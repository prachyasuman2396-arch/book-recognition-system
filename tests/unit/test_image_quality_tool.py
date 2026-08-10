from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.models.domain import QualityDecision
from app.tools.image_quality_tool import ImageQualityInput, ImageQualityTool


def _write_image(path: str, sharp: bool) -> None:
    if sharp:
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (350, 350), (255, 255, 255), -1)
        cv2.putText(img, "TITLE", (60, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    else:
        img = np.full((400, 400, 3), 128, dtype=np.uint8)
        img = cv2.GaussianBlur(img, (25, 25), 15)
    cv2.imwrite(path, img)


@pytest.mark.asyncio
async def test_sharp_high_contrast_image_scores_higher_than_blurry(tmp_settings, tmp_path):
    tool = ImageQualityTool(settings=tmp_settings)

    sharp_path = str(tmp_path / "sharp.jpg")
    blurry_path = str(tmp_path / "blurry.jpg")
    _write_image(sharp_path, sharp=True)
    _write_image(blurry_path, sharp=False)

    sharp_report = await tool.run(ImageQualityInput(detection_id="a", crop_path=sharp_path))
    blurry_report = await tool.run(ImageQualityInput(detection_id="b", crop_path=blurry_path))

    assert sharp_report.quality_score > blurry_report.quality_score
    assert blurry_report.decision in (QualityDecision.ENHANCE, QualityDecision.REJECT)


@pytest.mark.asyncio
async def test_missing_file_raises_tool_error(tmp_settings):
    from app.core.exceptions import ToolRetryExhaustedError

    tool = ImageQualityTool(settings=tmp_settings)
    with pytest.raises(ToolRetryExhaustedError):
        await tool.run(ImageQualityInput(detection_id="x", crop_path="/nonexistent/path.jpg"))
