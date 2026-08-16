"""Decoupled camera capture thread, bounded frame queue, frame sampling, and async DB logger."""
from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable

import cv2
import numpy as np

from config import Config
from modules.utils import get_logger


class FrameSampler:
    """Determine frame sampling interval to skip unnecessary frames while maintaining accuracy."""

    def __init__(self, skip_interval: int | None = None) -> None:
        self.skip_interval = max(1, skip_interval if skip_interval is not None else Config.FRAME_SKIP_INTERVAL)
        self.frame_count = 0

    def should_sample(self) -> bool:
        """Return True if current frame should undergo full detection & OCR."""
        self.frame_count += 1
        return (self.frame_count % self.skip_interval) == 0


class AsyncDBLogger:
    """Asynchronous background worker queue for database logging without blocking camera stream."""

    def __init__(self, maxsize: int = 100) -> None:
        self.logger = get_logger("db_async")
        self.queue: queue.Queue[Callable[[], None]] = queue.Queue(maxsize=maxsize)
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True, name="AsyncDBLogger")
        self.thread.start()

    def _worker_loop(self) -> None:
        while self.running:
            try:
                task = self.queue.get(timeout=1.0)
                try:
                    task()
                except Exception as exc:
                    self.logger.warning("Async DB task error: %s", exc)
                finally:
                    self.queue.task_done()
            except queue.Empty:
                continue

    def enqueue(self, task_func: Callable[[], None]) -> bool:
        """Enqueue a DB task without blocking. Drops if queue is full."""
        try:
            self.queue.put_nowait(task_func)
            return True
        except queue.Full:
            self.logger.warning("Async DB queue full (maxsize=%d), dropping task", self.queue.maxsize)
            return False

    def stop(self) -> None:
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)


class CameraCaptureThread:
    """Persistent background thread capturing camera/RTSP stream frames into a bounded queue."""

    def __init__(
        self,
        url: str | int,
        name: str = "Camera",
        max_queue_size: int | None = None,
        reconnect_interval: float | None = None,
    ) -> None:
        self.logger = get_logger("camera_stream")
        self.url = int(url) if str(url).isdigit() else str(url)
        self.name = name
        self.max_queue_size = max_queue_size if max_queue_size is not None else Config.MAX_FRAME_QUEUE_SIZE
        self.reconnect_interval = reconnect_interval if reconnect_interval is not None else Config.CAMERA_RECONNECT_INTERVAL

        self.frame_queue: queue.Queue[tuple[np.ndarray, float]] = queue.Queue(maxsize=self.max_queue_size)
        self.running = False
        self.cap: cv2.VideoCapture | None = None
        self.thread: threading.Thread | None = None
        
        self.fps = 0.0
        self.is_connected = False
        self.last_frame_time = 0.0
        self.frame_count = 0

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True, name=f"Cap_{self.name}")
        self.thread.start()

    def _connect(self) -> bool:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        try:
            self.cap = cv2.VideoCapture(self.url)
            # Set RTSP timeout property if supported
            if isinstance(self.url, str) and (self.url.startswith("rtsp://") or self.url.startswith("http://")):
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, Config.RTSP_TIMEOUT_MS)

            if self.cap.isOpened():
                self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
                self.is_connected = True
                self.logger.info("Camera connected: %s (FPS: %.1f)", self.name, self.fps)
                return True
            else:
                self.is_connected = False
                self.logger.warning("Camera connection failed: %s", self.name)
                return False
        except Exception as exc:
            self.is_connected = False
            self.logger.warning("Camera open error [%s]: %s", self.name, exc)
            return False

    def _capture_loop(self) -> None:
        while self.running:
            if not self.is_connected or self.cap is None or not self.cap.isOpened():
                if not self._connect():
                    time.sleep(self.reconnect_interval)
                    continue

            ret, frame = self.cap.read()
            now = time.perf_counter()

            if not ret or frame is None or frame.size == 0:
                self.is_connected = False
                self.logger.warning("Corrupted/empty frame from %s. Reconnecting...", self.name)
                time.sleep(self.reconnect_interval)
                continue

            self.frame_count += 1
            self.last_frame_time = now

            # Push frame to bounded queue, dropping oldest if queue full to prevent latency growth
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.frame_queue.put_nowait((frame, now))
            except queue.Full:
                pass

        self._cleanup()

    def get_latest_frame(self, timeout: float = 1.0) -> tuple[np.ndarray | None, float]:
        """Fetch the latest frame from the bounded queue."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None, 0.0

    def _cleanup(self) -> None:
        self.is_connected = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.logger.info("Camera capture stopped: %s", self.name)

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self._cleanup()


class CameraStreamManager:
    """Manage persistent camera capture threads across streams."""

    def __init__(self) -> None:
        self.sessions: dict[str, CameraCaptureThread] = {}
        self.db_logger = AsyncDBLogger()

    def get_stream(self, url: str | int, name: str = "Camera") -> CameraCaptureThread:
        key = str(url)
        if key not in self.sessions or not self.sessions[key].running:
            stream = CameraCaptureThread(url=url, name=name)
            stream.start()
            self.sessions[key] = stream
        return self.sessions[key]

    def stop_all(self) -> None:
        for stream in self.sessions.values():
            stream.stop()
        self.sessions.clear()
        self.db_logger.stop()
