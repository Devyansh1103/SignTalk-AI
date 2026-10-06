"""
SignTalk AI - Reusable Camera Abstraction
Provides camera acquisition, frame timestamping using monotonic timing,
thread-safe frame buffering, and resilient reconnection strategies.
"""

import time
import queue
import logging
import threading
from typing import Optional, Union, Tuple
import cv2
import numpy as np

from src.realtime.types import FramePacket

logger = logging.getLogger("SignTalk.RealTime.Camera")


class CameraError(Exception):
    """Base exception for camera capture failures."""
    pass


class CameraInitializationError(CameraError):
    """Raised when the camera device cannot be opened or initialized."""
    pass


class CameraReadError(CameraError):
    """Raised when frame capture fails or camera disconnects unexpectedly."""
    pass


class Camera:
    """
    Robust camera abstraction supporting physical webcams, virtual cameras,
    video streams, threaded frame acquisition, and monotonic timestamping.
    """

    def __init__(
        self,
        device_index: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        fps: float = 25.0,
        threaded: bool = False,
        buffer_size: int = 2,
        auto_reconnect: bool = True,
        max_reconnect_attempts: int = 3,
        reconnect_delay_sec: float = 1.0,
        loop_video: bool = False
    ):
        """
        Args:
            device_index: Hardware index (0, 1, ...) or video file path.
            width: Target horizontal resolution.
            height: Target vertical resolution.
            fps: Target capture frame rate.
            threaded: Whether to run capture loop in a dedicated background thread.
            buffer_size: Maximum queue capacity in threaded mode (keeps newest frames).
            auto_reconnect: Whether to attempt reconnection if read fails on device.
            max_reconnect_attempts: Max consecutive reconnection attempts.
            reconnect_delay_sec: Delay between reconnection attempts in seconds.
            loop_video: If True and source is a video file, loop back to start at EOF.
        """
        self.device_index = device_index
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = fps
        self.threaded = threaded
        self.buffer_size = buffer_size
        self.auto_reconnect = auto_reconnect
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay_sec = reconnect_delay_sec
        self.loop_video = loop_video

        self._cap: Optional[cv2.VideoCapture] = None
        self._is_running = False
        self._frame_counter = 0
        self._dropped_counter = 0
        self._actual_width = 0
        self._actual_height = 0
        self._actual_fps = 0.0

        # Threading infrastructure
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_queue: queue.Queue = queue.Queue(maxsize=self.buffer_size)
        self._lock = threading.Lock()

        # Timing tracking
        self._first_frame_ts: Optional[float] = None
        self._last_frame_ts: Optional[float] = None

    def start(self) -> "Camera":
        """Opens capture device and starts acquisition thread if enabled."""
        with self._lock:
            if self._is_running:
                logger.warning("Camera is already running.")
                return self

            self._open_capture()
            self._is_running = True
            self._stop_event.clear()

            if self.threaded:
                self._thread = threading.Thread(
                    target=self._capture_worker,
                    name="SignTalk-CameraWorker",
                    daemon=True
                )
                self._thread.start()
                logger.info(f"Started threaded camera capture worker on device={self.device_index}")
            else:
                logger.info(f"Started synchronous camera capture on device={self.device_index}")

            return self

    def _open_capture(self):
        """Initializes OpenCV VideoCapture instance and applies hardware properties."""
        logger.info(f"Initializing capture source: {self.device_index}")

        # On Windows, DirectShow backend (cv2.CAP_DSHOW) offers lower latency for USB webcams
        if isinstance(self.device_index, int):
            try:
                self._cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
                if not self._cap.isOpened():
                    self._cap = cv2.VideoCapture(self.device_index)
            except Exception:
                self._cap = cv2.VideoCapture(self.device_index)
        else:
            self._cap = cv2.VideoCapture(str(self.device_index))

        if not self._cap or not self._cap.isOpened():
            msg = (
                f"Camera unavailable (device={self.device_index}). "
                "Please check camera permissions, device connection, and device index."
            )
            logger.error(msg)
            raise CameraInitializationError(msg)

        # Configure hardware resolution and framerate if live device
        if isinstance(self.device_index, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
            self._cap.set(cv2.CAP_PROP_FPS, self.requested_fps)

        self._actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_val = self._cap.get(cv2.CAP_PROP_FPS)
        self._actual_fps = float(fps_val) if fps_val > 0 else float(self.requested_fps)

        logger.info(
            f"Capture initialized: {self._actual_width}x{self._actual_height} @ {self._actual_fps:.1f} FPS"
        )

    def _capture_worker(self):
        """Background thread worker continuously acquiring newest frames."""
        is_video_file = not isinstance(self.device_index, int)
        frame_interval = 1.0 / max(1.0, float(self.requested_fps))
        last_grab_time = time.perf_counter()

        while not self._stop_event.is_set():
            if is_video_file:
                # Pace video reading to simulate live camera frame rate
                now = time.perf_counter()
                elapsed = now - last_grab_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                last_grab_time = time.perf_counter()

            packet = self._grab_frame_internal()
            if packet is None:
                if self._stop_event.is_set():
                    break
                # Video file reached EOF
                if is_video_file and not self.loop_video:
                    break
                time.sleep(0.005)
                continue

            # Keep only the newest frame in the buffer (drop older frame if full)
            if self._frame_queue.full():
                try:
                    _ = self._frame_queue.get_nowait()
                    self._dropped_counter += 1
                except queue.Empty:
                    pass

            try:
                self._frame_queue.put_nowait(packet)
            except queue.Full:
                self._dropped_counter += 1

    def _grab_frame_internal(self) -> Optional[FramePacket]:
        """Reads a single frame from the underlying VideoCapture with reconnection support."""
        if self._cap is None or not self._cap.isOpened():
            return None

        t_mono = time.perf_counter()
        t_wall = time.time()

        ret, frame = self._cap.read()

        if not ret or frame is None:
            # Handle end of video file playback
            if not isinstance(self.device_index, int):
                if self.loop_video and self._cap is not None:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self._cap.read()
                    if not ret or frame is None:
                        return None
                else:
                    return None
            else:
                # Live camera read error - attempt auto-reconnect if enabled
                if self.auto_reconnect and not self._stop_event.is_set():
                    logger.warning("Live camera frame read failed. Attempting reconnection...")
                    success = self._attempt_reconnect()
                    if not success:
                        return None
                    ret, frame = self._cap.read()
                    if not ret or frame is None:
                        return None
                else:
                    return None

        h, w = frame.shape[:2]
        self._frame_counter += 1

        if self._first_frame_ts is None:
            self._first_frame_ts = t_mono
        self._last_frame_ts = t_mono

        return FramePacket(
            frame_id=self._frame_counter,
            timestamp=t_mono,
            capture_time=t_wall,
            image=frame,
            width=w,
            height=h,
            is_rgb=False,
            is_flipped=False
        )

    def _attempt_reconnect(self) -> bool:
        """Tries to reconnect to the camera device."""
        for attempt in range(1, self.max_reconnect_attempts + 1):
            if self._stop_event.is_set():
                return False
            logger.info(f"Reconnection attempt {attempt}/{self.max_reconnect_attempts}...")
            try:
                if self._cap is not None:
                    self._cap.release()
                time.sleep(self.reconnect_delay_sec)
                self._open_capture()
                logger.info("Successfully reconnected to camera!")
                return True
            except Exception as e:
                logger.warning(f"Reconnection attempt {attempt} failed: {e}")

        logger.error("Exceeded maximum reconnection attempts. Camera disconnected.")
        return False

    def read(self, timeout: float = 1.0) -> Optional[FramePacket]:
        """
        Reads the next available frame packet.

        In threaded mode: retrieves newest frame from thread queue.
        In synchronous mode: grabs directly from capture device.
        """
        if not self._is_running:
            return None

        if self.threaded:
            try:
                return self._frame_queue.get(timeout=timeout)
            except queue.Empty:
                return None
        else:
            return self._grab_frame_internal()

    def is_running(self) -> bool:
        """Checks if camera is actively capturing."""
        return self._is_running

    @property
    def frame_count(self) -> int:
        """Total number of successfully captured frames."""
        return self._frame_counter

    @property
    def dropped_count(self) -> int:
        """Total number of frames dropped due to buffer overflow in threaded mode."""
        return self._dropped_counter

    @property
    def actual_fps(self) -> float:
        """Nominal or measured device FPS."""
        return self._actual_fps

    @property
    def measured_fps(self) -> float:
        """Empirically calculated FPS based on monotonic frame timestamps."""
        if self._frame_counter < 2 or self._first_frame_ts is None or self._last_frame_ts is None:
            return 0.0
        elapsed = self._last_frame_ts - self._first_frame_ts
        return (self._frame_counter - 1) / elapsed if elapsed > 0 else 0.0

    @property
    def resolution(self) -> Tuple[int, int]:
        """Actual capture resolution (width, height)."""
        return self._actual_width, self._actual_height

    def stop(self):
        """Stops capture and releases all hardware resources cleanly."""
        with self._lock:
            if not self._is_running:
                return

            self._is_running = False
            self._stop_event.set()

            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=2.0)
                self._thread = None

            if self._cap is not None:
                self._cap.release()
                self._cap = None

            logger.info("Camera stopped and hardware resources released.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
