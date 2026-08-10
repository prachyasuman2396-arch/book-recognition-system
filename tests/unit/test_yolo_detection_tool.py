from __future__ import annotations

import pytest

from app.tools.yolo_detection_tool import YOLODetectionTool, YoloDetectionInput


@pytest.mark.asyncio
async def test_fallback_detector_returns_one_full_image_box(tmp_settings, sample_image_path):
    tool = YOLODetectionTool(settings=tmp_settings)
    detections = await tool.run(
        YoloDetectionInput(image_path=sample_image_path, request_id="req-1")
    )

    assert len(detections) == 1
    detection = detections[0]
    assert detection.confidence == pytest.approx(0.5)
    assert detection.bbox.width > 0
    assert detection.bbox.height > 0
    from pathlib import Path

    assert Path(detection.crop_path).exists()


@pytest.mark.asyncio
async def test_missing_image_raises(tmp_settings):
    from app.core.exceptions import ToolRetryExhaustedError

    tool = YOLODetectionTool(settings=tmp_settings)
    with pytest.raises(ToolRetryExhaustedError):
        await tool.run(YoloDetectionInput(image_path="/no/such/file.jpg", request_id="req-2"))
