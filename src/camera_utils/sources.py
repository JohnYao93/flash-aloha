"""Camera sources for task calibration and previews."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import time
import numpy as np

@dataclass
class CameraRead:
    frames: dict[str, np.ndarray]
    frame_numbers: dict[str, int]
    timestamps: dict[str, float]


class CameraSource:
    """Read frames from shared-memory camera_server or direct V4L2 capture."""

    def __init__(
        self,
        camera_names: list[str],
        use_camera_server: bool = True,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self.camera_names = list(camera_names)
        self.use_camera_server = bool(use_camera_server)
        self.width = width
        self.height = height
        self.readers: dict[str, Any] = {}
        self.captures: dict[str, Any] = {}

    @staticmethod
    def short_name(camera_name: str) -> str:
        return camera_name.replace("video_", "", 1)

    def open(self) -> None:
        if self.use_camera_server:
            from camera_utils.capture import SharedFrameReader

            for camera_name in self.camera_names:
                short = self.short_name(camera_name)
                reader = SharedFrameReader(short, width=self.width, height=self.height)
                for attempt in range(50):
                    if reader.connect():
                        break
                    time.sleep(0.1)
                else:
                    raise RuntimeError(f"Camera server shared memory not available for {camera_name}")
                self.readers[camera_name] = reader
        else:
            from camera_utils.capture import init_camera

            for camera_name in self.camera_names:
                cap = init_camera(f"/dev/{camera_name}")
                if cap is None or not cap.isOpened():
                    raise RuntimeError(f"Failed to open /dev/{camera_name}")
                self.captures[camera_name] = cap

    def read(self) -> CameraRead:
        frames: dict[str, np.ndarray] = {}
        frame_numbers: dict[str, int] = {}
        timestamps: dict[str, float] = {}
        if self.use_camera_server:
            for camera_name, reader in self.readers.items():
                frame, frame_number, timestamp = reader.read_frame()
                if frame is None:
                    raise RuntimeError(f"No frame available from {camera_name}")
                frames[camera_name] = frame
                frame_numbers[camera_name] = int(frame_number)
                timestamps[camera_name] = float(timestamp)
        else:
            for camera_name, cap in self.captures.items():
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Failed to read frame from {camera_name}")
                frames[camera_name] = frame
                frame_numbers[camera_name] = -1
                timestamps[camera_name] = time.time()
        return CameraRead(frames=frames, frame_numbers=frame_numbers, timestamps=timestamps)

    def close(self) -> None:
        for reader in self.readers.values():
            reader.close()
        for cap in self.captures.values():
            cap.release()
        self.readers.clear()
        self.captures.clear()


