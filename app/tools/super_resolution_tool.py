"""SuperResolutionTool -- enhances low-quality crops via Real-ESRGAN.

Never overwrites the original crop; always writes to a new path under
`ENHANCED_DIR`. Falls back to a high-quality Lanczos upscale (via Pillow)
if the `realesrgan` package or weights aren't available in the current
environment, keeping the pipeline runnable end-to-end without a GPU.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from app.config import Settings, get_settings
from app.core.exceptions import ToolError
from app.models.domain import EnhancedImage
from app.tools.base_tool import BaseTool


@dataclass
class SuperResolutionInput:
    detection_id: str
    crop_path: str
    request_id: str


class SuperResolutionTool(BaseTool[SuperResolutionInput, EnhancedImage]):
    name = "super_resolution_tool"

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__()
        self.settings = settings or get_settings()
        self.timeout_seconds = 60.0
        self.max_retries = 2
        self._upsampler = None

    def _load_upsampler(self):
        if self._upsampler is not None:
            return self._upsampler
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet  # type: ignore
            from realesrgan import RealESRGANer  # type: ignore

            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32,
                             scale=self.settings.SUPER_RES_SCALE)
            self._upsampler = RealESRGANer(
                scale=self.settings.SUPER_RES_SCALE,
                model_path=f"weights/{self.settings.SUPER_RES_MODEL_NAME}.pth",
                model=model,
                tile=self.settings.SUPER_RES_TILE,
                device=self.settings.SUPER_RES_DEVICE,
            )
            return self._upsampler
        except (ImportError, FileNotFoundError):
            self.logger.warning(
                "Real-ESRGAN unavailable; falling back to Lanczos upscale"
            )
            return None

    def warm_up(self) -> None:
        """Synchronously load weights. Call from a worker thread only
        (see `Container.warm_up`) -- never from the event loop directly."""
        self._load_upsampler()

    async def run(self, payload: SuperResolutionInput) -> EnhancedImage:
        return await self._execute(self._enhance, payload)

    async def _enhance(self, payload: SuperResolutionInput) -> EnhancedImage:
        # See YOLODetectionTool._detect: this is blocking, CPU/GPU-bound
        # work (model load, inference, disk I/O) and must not run directly
        # on the event loop.
        return await asyncio.to_thread(self._enhance_sync, payload)

    def _enhance_sync(self, payload: SuperResolutionInput) -> EnhancedImage:
        crop_path = Path(payload.crop_path)
        if not crop_path.exists():
            raise ToolError(f"Crop not found for enhancement: {crop_path}")

        enhanced_dir = Path(self.settings.ENHANCED_DIR) / payload.request_id
        enhanced_dir.mkdir(parents=True, exist_ok=True)
        enhanced_path = enhanced_dir / f"{payload.detection_id}_enhanced.jpg"

        upsampler = self._load_upsampler()
        model_used = self.settings.SUPER_RES_MODEL_NAME

        if upsampler is not None:
            import cv2

            img = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
            output, _ = upsampler.enhance(img, outscale=self.settings.SUPER_RES_SCALE)
            cv2.imwrite(str(enhanced_path), output)
        else:
            model_used = "lanczos_fallback"
            with Image.open(crop_path) as img:
                img = img.convert("RGB")
                new_size = (
                    img.width * self.settings.SUPER_RES_SCALE,
                    img.height * self.settings.SUPER_RES_SCALE,
                )
                upscaled = img.resize(new_size, Image.LANCZOS)
                upscaled.save(enhanced_path, format="JPEG", quality=95)

        return EnhancedImage(
            detection_id=payload.detection_id,
            original_crop_path=str(crop_path),
            enhanced_path=str(enhanced_path),
            scale_factor=self.settings.SUPER_RES_SCALE,
            model_used=model_used,
        )
