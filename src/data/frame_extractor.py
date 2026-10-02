"""
SignTalk AI - Frame Extractor
Extracts, resamples, and indexes video frames safely for landmark extraction.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Generator


class FrameExtractionError(Exception):
    """Raised when frame extraction fails."""
    pass


class FrameExtractor:
    """
    Extracts frames from video files or individual image files with configurable
    sampling rate, target FPS, resizing, and precise timestamp tracking.
    """

    def __init__(
        self,
        target_fps: float = 25.0,
        frame_sampling_rate: int = 1,
        image_size: Optional[Tuple[int, int]] = None,  # (width, height)
        uniform_resampling: bool = True
    ):
        self.target_fps = float(target_fps)
        self.frame_sampling_rate = max(1, int(frame_sampling_rate))
        self.image_size = image_size
        self.uniform_resampling = uniform_resampling

    def extract_from_video(self, video_path: str) -> Dict[str, Any]:
        """
        Safely extracts frames and metadata from a video file.

        Args:
            video_path: Absolute or relative path to the input video.

        Returns:
            Dictionary containing:
                - 'frames': List of RGB numpy arrays (H, W, 3)
                - 'metadata': Dictionary of video properties and frame indexes
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file does not exist: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FrameExtractionError(f"OpenCV failed to open video: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Handle corrupt or unreadable video header
        if total_raw_frames <= 0 or original_fps <= 0:
            # Fallback: read frame by frame to count
            raw_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                raw_frames.append(frame)
            cap.release()
            total_raw_frames = len(raw_frames)
            original_fps = original_fps if original_fps > 0 else self.target_fps
            frames_source = raw_frames
        else:
            frames_source = None

        if total_raw_frames == 0:
            if cap.isOpened():
                cap.release()
            raise FrameExtractionError(f"Video contains 0 decodable frames: {video_path}")

        # Compute sampling indices based on uniform resampling or fixed stride
        if self.uniform_resampling and abs(original_fps - self.target_fps) > 0.5:
            # Calculate target number of frames preserving duration
            duration_sec = total_raw_frames / original_fps
            target_frame_count = max(1, int(round(duration_sec * self.target_fps)))
            sample_indices = np.linspace(0, total_raw_frames - 1, target_frame_count, dtype=int)
            sample_indices = np.unique(sample_indices).tolist()
        else:
            sample_indices = list(range(0, total_raw_frames, self.frame_sampling_rate))

        sample_set = set(sample_indices)
        extracted_frames: List[np.ndarray] = []
        frame_records: List[Dict[str, Any]] = []

        if frames_source is not None:
            for idx in sample_indices:
                bgr_frame = frames_source[idx]
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                if self.image_size:
                    rgb_frame = cv2.resize(rgb_frame, self.image_size, interpolation=cv2.INTER_LINEAR)
                timestamp_sec = idx / original_fps
                extracted_frames.append(rgb_frame)
                frame_records.append({
                    "frame_index": len(extracted_frames) - 1,
                    "original_frame_index": idx,
                    "timestamp_sec": round(timestamp_sec, 4),
                    "timestamp_ms": round(timestamp_sec * 1000.0, 2)
                })
        else:
            curr_idx = 0
            while True:
                ret, bgr_frame = cap.read()
                if not ret:
                    break
                if curr_idx in sample_set:
                    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                    if self.image_size:
                        rgb_frame = cv2.resize(rgb_frame, self.image_size, interpolation=cv2.INTER_LINEAR)
                    timestamp_sec = curr_idx / original_fps
                    extracted_frames.append(rgb_frame)
                    frame_records.append({
                        "frame_index": len(extracted_frames) - 1,
                        "original_frame_index": curr_idx,
                        "timestamp_sec": round(timestamp_sec, 4),
                        "timestamp_ms": round(timestamp_sec * 1000.0, 2)
                    })
                curr_idx += 1
            cap.release()

        if len(extracted_frames) == 0:
            raise FrameExtractionError(f"No frames were sampled from video: {video_path}")

        h, w = extracted_frames[0].shape[:2]
        effective_fps = len(extracted_frames) / (total_raw_frames / original_fps) if total_raw_frames > 0 else self.target_fps

        metadata = {
            "video_path": video_path,
            "original_fps": round(original_fps, 2),
            "target_fps": self.target_fps,
            "effective_fps": round(effective_fps, 2),
            "original_frame_count": total_raw_frames,
            "extracted_frame_count": len(extracted_frames),
            "original_resolution": [orig_width, orig_height],
            "extracted_resolution": [w, h],
            "duration_sec": round(total_raw_frames / original_fps, 3),
            "frame_records": frame_records
        }

        return {
            "frames": extracted_frames,
            "metadata": metadata
        }

    def extract_from_image(self, image_path: str) -> Dict[str, Any]:
        """
        Extracts a single frame from an image file.

        Args:
            image_path: Path to the image file.

        Returns:
            Dictionary containing 'frames' (single element list) and 'metadata'.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        bgr_frame = cv2.imread(image_path)
        if bgr_frame is None:
            raise FrameExtractionError(f"OpenCV failed to read image: {image_path}")

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        if self.image_size:
            rgb_frame = cv2.resize(rgb_frame, self.image_size, interpolation=cv2.INTER_LINEAR)

        h, w = rgb_frame.shape[:2]
        metadata = {
            "image_path": image_path,
            "original_fps": self.target_fps,
            "target_fps": self.target_fps,
            "effective_fps": self.target_fps,
            "original_frame_count": 1,
            "extracted_frame_count": 1,
            "original_resolution": [bgr_frame.shape[1], bgr_frame.shape[0]],
            "extracted_resolution": [w, h],
            "duration_sec": round(1.0 / self.target_fps, 3),
            "frame_records": [{
                "frame_index": 0,
                "original_frame_index": 0,
                "timestamp_sec": 0.0,
                "timestamp_ms": 0.0
            }]
        }

        return {
            "frames": [rgb_frame],
            "metadata": metadata
        }
