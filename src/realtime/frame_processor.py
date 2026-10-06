"""
SignTalk AI - Real-Time Frame Preprocessor
Applies standard vision transforms to prepare raw camera frames for
MediaPipe landmark extraction and UI preview rendering:
  - BGR -> RGB color space conversion
  - Optional horizontal mirroring (natural signer preview)
  - Optional resolution standardization
  - Input integrity and dimension validation
"""

import time
from typing import Optional, Tuple
import cv2
import numpy as np

from src.realtime.types import FramePacket
from src.realtime.realtime_config import ProcessingConfig


class FrameProcessingError(Exception):
    """Raised when frame preprocessing fails."""
    pass


class FrameProcessor:
    """
    Standardized real-time frame preprocessor ensuring pixel formats and
    orientations match training pipeline assumptions.
    """

    def __init__(
        self,
        config: Optional[ProcessingConfig] = None,
        flip_horizontal: bool = True,
        color_format: str = "RGB",
        target_width: Optional[int] = None,
        target_height: Optional[int] = None
    ):
        if config is not None:
            self.flip_horizontal = config.flip_horizontal
            self.color_format = config.color_format.upper()
            self.target_width = config.target_width
            self.target_height = config.target_height
        else:
            self.flip_horizontal = flip_horizontal
            self.color_format = color_format.upper()
            self.target_width = target_width
            self.target_height = target_height

    def process(self, packet: FramePacket) -> FramePacket:
        """
        Processes a raw captured FramePacket:
        1. Validates array dimensions and data type.
        2. Applies optional horizontal flip.
        3. Converts BGR to RGB (if not already RGB).
        4. Resizes if target dimensions differ from current resolution.

        Returns:
            A new FramePacket ready for MediaPipe consumption.
        """
        img = packet.image
        if img is None or img.size == 0:
            raise FrameProcessingError(f"Received empty image array in frame {packet.frame_id}")

        if img.dtype != np.uint8:
            raise FrameProcessingError(f"Expected uint8 image, got {img.dtype}")

        if img.ndim != 3 or img.shape[2] != 3:
            raise FrameProcessingError(f"Expected 3-channel image (H, W, 3), got shape {img.shape}")

        processed_img = img

        # 1. Horizontal Flip (Mirroring for natural webcam signing)
        is_flipped = packet.is_flipped
        if self.flip_horizontal and not packet.is_flipped:
            processed_img = cv2.flip(processed_img, 1)
            is_flipped = True

        # 2. Resizing if required
        h, w = processed_img.shape[:2]
        if self.target_width is not None and self.target_height is not None:
            if w != self.target_width or h != self.target_height:
                processed_img = cv2.resize(
                    processed_img,
                    (self.target_width, self.target_height),
                    interpolation=cv2.INTER_LINEAR
                )
                h, w = self.target_height, self.target_width

        # 3. Color Space Conversion (OpenCV captures in BGR -> MediaPipe requires RGB)
        is_rgb = packet.is_rgb
        if self.color_format == "RGB" and not packet.is_rgb:
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            is_rgb = True
        elif self.color_format == "BGR" and packet.is_rgb:
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
            is_rgb = False

        return FramePacket(
            frame_id=packet.frame_id,
            timestamp=packet.timestamp,
            capture_time=packet.capture_time,
            image=processed_img,
            width=w,
            height=h,
            is_rgb=is_rgb,
            is_flipped=is_flipped
        )

    @staticmethod
    def to_bgr_for_preview(image: np.ndarray, is_rgb: bool = True) -> np.ndarray:
        """Helper to convert RGB frame back to BGR for OpenCV GUI rendering."""
        if is_rgb:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image.copy()
