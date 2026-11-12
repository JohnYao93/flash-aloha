"""
Single-camera capture server that logs frames with robologger.

Run one process per camera for lower latency and simpler scheduling:
- Process 1: --camera left
- Process 2: --camera overhead
- Process 3: --camera right
- Process 4: --camera torso

The shared DistributedRecorder backend connects to all per-camera logger
endpoints and starts/stops them together.
"""

# Support direct execution as well as package imports.
import sys as _path_sys
from pathlib import Path as _Path
_ACT_ROOT = next(p for p in _Path(__file__).resolve().parents if p.name == "src")
if str(_ACT_ROOT) not in _path_sys.path:
    _path_sys.path.insert(0, str(_ACT_ROOT))

import argparse
from collections import deque
import ctypes
from dataclasses import dataclass
import errno
import fcntl
import json
import mmap
from multiprocessing import shared_memory
import os
from pathlib import Path
import re
import select
import signal
import struct
import sys
import threading
import time

import cv2
import numpy as np

from camera_utils.camera_timing import driver_timestamp, publish_shared_clock_contract
from contracts import capture_gc


DEFAULT_CONTROL_SETTINGS = {
    "brightness": 0,
    "contrast": 32,
    "saturation": 90,
    "hue": 0,
    "auto_wb": 1,
    "gamma": 100,
    "gain": 30,
    "power_line_frequency": 2,
    "wb_temperature": 3615,
    "sharpness": 3,
    "backlight_compensation": 1,
    "auto_exposure": 1,
    "exposure_absolute": 156,
    "exposure_dynamic_framerate": 0,
    "autofocus": 0,
    "focus": 20,
}

TUNED_CONTROL_NAMES = tuple(DEFAULT_CONTROL_SETTINGS)



@dataclass(frozen=True)
class CameraConfig:
    name: str
    device_path: str
    width: int
    height: int
    fps: float
    settings: dict[str, int]
    params_path: str
    loaded_from_params: bool


def get_camera_params_npz_path() -> str:
    env_path = os.environ.get("FLASH_ALOHA_CAMERA_PARAMS")
    if env_path:
        return env_path

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    calibration_path = os.path.join(repo_root, "calibration", "camera_params.npz")
    if os.path.exists(calibration_path):
        return calibration_path
    return os.path.join(repo_root, "camera_params.npz")


def camera_match_keys(name: str, device_path: str) -> set[str]:
    keys = {str(name), str(device_path)}
    basename = os.path.basename(str(device_path))
    if basename:
        keys.add(basename)
        keys.add(basename.replace("video_", ""))
    if str(name).startswith("video_"):
        keys.add(str(name).replace("video_", ""))
    else:
        keys.add(f"video_{name}")
    return {key for key in keys if key}


def _scalar_or_index(values: np.ndarray, idx: int) -> int | None:
    values = np.asarray(values)
    if values.ndim == 0:
        return int(values)
    if idx < values.shape[0]:
        return int(values[idx])
    return None


def load_camera_param_records(path: str | None = None, *, strict: bool = False) -> list[dict]:
    path = path or get_camera_params_npz_path()
    if not os.path.exists(path):
        return []

    records = []
    try:
        with np.load(path, allow_pickle=False) as params:
            camera_names = [str(name) for name in params.get("camera_names", [])]
            devices = [str(device) for device in params.get("devices", [])]
            count = max(len(camera_names), len(devices))
            if count == 0:
                return []

            param_names = [str(name) for name in params.get("param_names", [])]
            values = np.asarray(params["values"]) if "values" in params else None

            for idx in range(count):
                name = camera_names[idx] if idx < len(camera_names) else f"cam{idx}"
                device = devices[idx] if idx < len(devices) else f"/dev/video_{name}"

                settings = {}
                for key in TUNED_CONTROL_NAMES:
                    value = None
                    if key in params:
                        value = _scalar_or_index(params[key], idx)
                    elif values is not None and key in param_names and idx < values.shape[0]:
                        value = int(values[idx, param_names.index(key)])
                    if value is not None:
                        settings[key] = int(value)

                width = _scalar_or_index(params["widths"], idx) if "widths" in params else None
                height = _scalar_or_index(params["heights"], idx) if "heights" in params else None
                fps = _scalar_or_index(params["fps_values"], idx) if "fps_values" in params else None
                if width is None and "width" in params:
                    width = _scalar_or_index(params["width"], idx)
                if height is None and "height" in params:
                    height = _scalar_or_index(params["height"], idx)
                if fps is None and "fps" in params:
                    fps = _scalar_or_index(params["fps"], idx)

                records.append(
                    {
                        "name": name,
                        "device": device,
                        "settings": settings,
                        "mode": (width, height, fps) if None not in (width, height, fps) else None,
                    }
                )
    except Exception as exc:
        if strict:
            raise ValueError(f"Invalid camera settings file {path}: {exc}") from exc
        print(f"[WARN] Failed to load camera params from {path}: {exc}")
    return records


def find_camera_param_record(name: str, device_path: str, path: str | None = None) -> dict | None:
    keys = camera_match_keys(name, device_path)
    for record in load_camera_param_records(path):
        if keys & camera_match_keys(record["name"], record["device"]):
            return record
    return None


def resolve_camera_config(
    name: str,
    device_path: str,
    width_override: int | None = None,
    height_override: int | None = None,
    fps_override: float | None = None,
    fallback_width: int = 1280,
    fallback_height: int = 720,
    fallback_fps: float = 50,
    params_path: str | None = None,
) -> CameraConfig:
    params_path = params_path or get_camera_params_npz_path()
    record = find_camera_param_record(name, device_path, params_path)
    saved_mode = record.get("mode") if record else None
    saved_settings = record.get("settings", {}) if record else {}

    saved_width, saved_height, saved_fps = saved_mode or (None, None, None)
    settings = dict(DEFAULT_CONTROL_SETTINGS)
    settings.update({key: int(value) for key, value in saved_settings.items()})

    return CameraConfig(
        name=name,
        device_path=device_path,
        width=int(width_override if width_override is not None else (saved_width or fallback_width)),
        height=int(height_override if height_override is not None else (saved_height or fallback_height)),
        fps=float(fps_override if fps_override is not None else (saved_fps or fallback_fps)),
        settings=settings,
        params_path=params_path,
        loaded_from_params=record is not None,
    )

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP = 1
V4L2_FIELD_ANY = 0
V4L2_CTRL_FLAG_NEXT_CTRL = 0x80000000
V4L2_CTRL_FLAG_DISABLED = 0x0002


def _fourcc(code: str) -> int:
    return (
        ord(code[0])
        | (ord(code[1]) << 8)
        | (ord(code[2]) << 16)
        | (ord(code[3]) << 24)
    )


V4L2_PIX_FMT_MJPEG = _fourcc("MJPG")


def _ioc(direction: int, type_: str, nr: int, size: int) -> int:
    return (direction << 30) | (size << 16) | (ord(type_) << 8) | nr


def _ior(type_: str, nr: int, struct_type) -> int:
    return _ioc(2, type_, nr, ctypes.sizeof(struct_type))


def _iow(type_: str, nr: int, struct_type) -> int:
    return _ioc(1, type_, nr, ctypes.sizeof(struct_type))


def _iowr(type_: str, nr: int, struct_type) -> int:
    return _ioc(3, type_, nr, ctypes.sizeof(struct_type))


class V4L2Fract(ctypes.Structure):
    _fields_ = [("numerator", ctypes.c_uint32), ("denominator", ctypes.c_uint32)]


class V4L2Timeval(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class V4L2Timecode(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("frames", ctypes.c_uint8),
        ("seconds", ctypes.c_uint8),
        ("minutes", ctypes.c_uint8),
        ("hours", ctypes.c_uint8),
        ("userbits", ctypes.c_uint8 * 4),
    ]


class V4L2Capability(ctypes.Structure):
    _fields_ = [
        ("driver", ctypes.c_char * 16),
        ("card", ctypes.c_char * 32),
        ("bus_info", ctypes.c_char * 32),
        ("version", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("device_caps", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
    ]


class V4L2PixFormat(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("pixelformat", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("bytesperline", ctypes.c_uint32),
        ("sizeimage", ctypes.c_uint32),
        ("colorspace", ctypes.c_uint32),
        ("priv", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("ycbcr_enc", ctypes.c_uint32),
        ("quantization", ctypes.c_uint32),
        ("xfer_func", ctypes.c_uint32),
    ]


class V4L2FormatUnion(ctypes.Union):
    _fields_ = [
        ("pix", V4L2PixFormat),
        ("raw_data", ctypes.c_uint8 * 200),
        ("_align", ctypes.c_void_p),
    ]


class V4L2Format(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("fmt", V4L2FormatUnion)]


class V4L2CaptureParm(ctypes.Structure):
    _fields_ = [
        ("capability", ctypes.c_uint32),
        ("capturemode", ctypes.c_uint32),
        ("timeperframe", V4L2Fract),
        ("extendedmode", ctypes.c_uint32),
        ("readbuffers", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 4),
    ]


class V4L2StreamParmUnion(ctypes.Union):
    _fields_ = [("capture", V4L2CaptureParm), ("raw_data", ctypes.c_uint8 * 200)]


class V4L2StreamParm(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("parm", V4L2StreamParmUnion)]


class V4L2RequestBuffers(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("flags", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 3),
    ]


class V4L2BufferM(ctypes.Union):
    _fields_ = [
        ("offset", ctypes.c_uint32),
        ("userptr", ctypes.c_ulong),
        ("planes", ctypes.c_void_p),
        ("fd", ctypes.c_int32),
    ]


class V4L2BufferRequestUnion(ctypes.Union):
    _fields_ = [("request_fd", ctypes.c_int32), ("reserved", ctypes.c_uint32)]


class V4L2Buffer(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("bytesused", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("timestamp", V4L2Timeval),
        ("timecode", V4L2Timecode),
        ("sequence", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("m", V4L2BufferM),
        ("length", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
        ("request", V4L2BufferRequestUnion),
    ]


class V4L2QueryCtrl(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("name", ctypes.c_char * 32),
        ("minimum", ctypes.c_int32),
        ("maximum", ctypes.c_int32),
        ("step", ctypes.c_int32),
        ("default_value", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 2),
    ]


class V4L2Control(ctypes.Structure):
    _fields_ = [("id", ctypes.c_uint32), ("value", ctypes.c_int32)]


VIDIOC_QUERYCAP = _ior("V", 0, V4L2Capability)
VIDIOC_S_FMT = _iowr("V", 5, V4L2Format)
VIDIOC_REQBUFS = _iowr("V", 8, V4L2RequestBuffers)
VIDIOC_QUERYBUF = _iowr("V", 9, V4L2Buffer)
VIDIOC_QBUF = _iowr("V", 15, V4L2Buffer)
VIDIOC_DQBUF = _iowr("V", 17, V4L2Buffer)
VIDIOC_STREAMON = _iow("V", 18, ctypes.c_int)
VIDIOC_STREAMOFF = _iow("V", 19, ctypes.c_int)
VIDIOC_S_PARM = _iowr("V", 22, V4L2StreamParm)
VIDIOC_QUERYCTRL = _iowr("V", 36, V4L2QueryCtrl)
VIDIOC_S_CTRL = _iowr("V", 28, V4L2Control)


CONTROL_ALIASES = {
    "brightness": ("brightness",),
    "contrast": ("contrast",),
    "saturation": ("saturation",),
    "hue": ("hue",),
    "auto_wb": (
        "white_balance_automatic",
        "auto_white_balance",
        "white_balance_temperature_auto",
    ),
    "gamma": ("gamma",),
    "gain": ("gain",),
    "power_line_frequency": ("power_line_frequency",),
    "wb_temperature": ("white_balance_temperature", "wb_temperature"),
    "sharpness": ("sharpness",),
    "backlight_compensation": ("backlight_compensation",),
    "auto_exposure": ("auto_exposure", "exposure_auto"),
    "exposure_absolute": (
        "exposure_time_absolute",
        "exposure_absolute",
        "exposure",
    ),
    "exposure_dynamic_framerate": ("exposure_dynamic_framerate",),
    "autofocus": ("focus_auto", "focus_automatic", "autofocus"),
    "focus": ("focus_absolute", "focus"),
}


def _normalize_control_name(name: bytes | str) -> str:
    if isinstance(name, bytes):
        name = name.split(b"\0", maxsplit=1)[0].decode(errors="ignore")
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if normalized == "exposure_auto":
        return "auto_exposure"
    return normalized


def _xioctl(fd: int, request: int, arg) -> None:
    while True:
        try:
            fcntl.ioctl(fd, request, arg, True)
            return
        except InterruptedError:
            continue


class V4L2MmapCamera:
    """Native V4L2 mmap MJPG reader that returns decoded BGR frames."""

    def __init__(
        self,
        device_path: str,
        width: int,
        height: int,
        fps: float,
        controls: dict[str, int] | None = None,
        buffer_count: int = 4,
        read_timeout_s: float = 1.0,
    ):
        self.device_path = device_path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.controls = dict(controls or {})
        self.buffer_count = max(2, int(buffer_count))
        self.read_timeout_s = max(0.001, float(read_timeout_s))
        self.fd = -1
        self.buffers: list[mmap.mmap] = []
        self.streaming = False
        self.last_timestamp = 0.0
        self.last_sequence = -1
        self.last_buffer_flags = 0
        self.last_dequeue_time = 0.0
        self.last_decode_time = 0.0
        self.driver = ""
        self.card = ""
        self.control_ids: dict[str, int] = {}
        self.applied_controls: dict[str, int] = {}
        self.missing_controls: list[str] = []

    def open(self) -> None:
        if not os.path.exists(self.device_path):
            raise FileNotFoundError(f"{self.device_path} does not exist")

        self.fd = os.open(self.device_path, os.O_RDWR | os.O_NONBLOCK)
        try:
            self._query_capability()
            self.control_ids = self._enumerate_control_ids()
            self._set_control_by_alias("exposure_dynamic_framerate", 0)
            self._set_format()
            self._set_fps()
            self._apply_controls()
            self._init_mmap()
            self._start_streaming()
        except Exception:
            self.release()
            raise

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        if self.fd < 0 or not self.streaming:
            return False, None, 0.0

        ready, _, _ = select.select([self.fd], [], [], self.read_timeout_s)
        if not ready:
            return False, None, 0.0

        buf = V4L2Buffer()
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = V4L2_MEMORY_MMAP
        try:
            _xioctl(self.fd, VIDIOC_DQBUF, buf)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EINTR):
                return False, None, 0.0
            raise

        try:
            dequeue_time = time.monotonic()
            data = bytes(self.buffers[buf.index][: buf.bytesused])
            encoded = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            decode_time = time.monotonic()
            timestamp = float(buf.timestamp.tv_sec) + float(buf.timestamp.tv_usec) / 1_000_000.0
            timestamp = driver_timestamp(timestamp, int(buf.flags))
            self.last_timestamp = timestamp
            self.last_sequence = int(buf.sequence)
            self.last_buffer_flags = int(buf.flags)
            self.last_dequeue_time = dequeue_time
            self.last_decode_time = decode_time
            if frame is None:
                return False, None, timestamp
            return True, frame, timestamp
        finally:
            _xioctl(self.fd, VIDIOC_QBUF, buf)

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop_id == cv2.CAP_PROP_FPS:
            return float(self.fps)
        if prop_id == cv2.CAP_PROP_POS_MSEC:
            return float(self.last_timestamp) * 1000.0
        return 0.0

    def release(self) -> None:
        if self.fd >= 0 and self.streaming:
            buf_type = ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE)
            try:
                fcntl.ioctl(self.fd, VIDIOC_STREAMOFF, buf_type)
            except OSError:
                pass
            self.streaming = False

        for mapped in self.buffers:
            try:
                mapped.close()
            except Exception:
                pass
        self.buffers.clear()

        if self.fd >= 0:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = -1

    def _query_capability(self) -> None:
        cap = V4L2Capability()
        _xioctl(self.fd, VIDIOC_QUERYCAP, cap)
        self.driver = cap.driver.split(b"\0", maxsplit=1)[0].decode(errors="ignore")
        self.card = cap.card.split(b"\0", maxsplit=1)[0].decode(errors="ignore")

    def _set_format(self) -> None:
        fmt = V4L2Format()
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        fmt.fmt.pix.width = self.width
        fmt.fmt.pix.height = self.height
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG
        fmt.fmt.pix.field = V4L2_FIELD_ANY
        _xioctl(self.fd, VIDIOC_S_FMT, fmt)
        self.width = int(fmt.fmt.pix.width)
        self.height = int(fmt.fmt.pix.height)

    def _set_fps(self) -> None:
        parm = V4L2StreamParm()
        parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        parm.parm.capture.timeperframe.numerator = 1
        parm.parm.capture.timeperframe.denominator = max(1, int(round(self.fps)))
        _xioctl(self.fd, VIDIOC_S_PARM, parm)
        numerator = int(parm.parm.capture.timeperframe.numerator)
        denominator = int(parm.parm.capture.timeperframe.denominator)
        if numerator > 0 and denominator > 0:
            self.fps = float(denominator) / float(numerator)

    def _init_mmap(self) -> None:
        req = V4L2RequestBuffers()
        req.count = self.buffer_count
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        req.memory = V4L2_MEMORY_MMAP
        _xioctl(self.fd, VIDIOC_REQBUFS, req)
        if req.count < 2:
            raise RuntimeError(f"{self.device_path} returned too few mmap buffers: {req.count}")

        for idx in range(req.count):
            buf = V4L2Buffer()
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            buf.memory = V4L2_MEMORY_MMAP
            buf.index = idx
            _xioctl(self.fd, VIDIOC_QUERYBUF, buf)
            mapped = mmap.mmap(
                self.fd,
                int(buf.length),
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
                offset=int(buf.m.offset),
            )
            self.buffers.append(mapped)
            _xioctl(self.fd, VIDIOC_QBUF, buf)

    def _start_streaming(self) -> None:
        buf_type = ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE)
        _xioctl(self.fd, VIDIOC_STREAMON, buf_type)
        self.streaming = True

    def _enumerate_control_ids(self) -> dict[str, int]:
        controls = {}
        query = V4L2QueryCtrl()
        query.id = V4L2_CTRL_FLAG_NEXT_CTRL
        while True:
            try:
                _xioctl(self.fd, VIDIOC_QUERYCTRL, query)
            except OSError as exc:
                if exc.errno == errno.EINVAL:
                    break
                raise
            if not query.flags & V4L2_CTRL_FLAG_DISABLED:
                controls[_normalize_control_name(query.name)] = int(query.id)
            query.id |= V4L2_CTRL_FLAG_NEXT_CTRL
        return controls

    def _set_control_by_alias(self, name: str, value: int) -> bool:
        aliases = CONTROL_ALIASES.get(name, (name,))
        for alias in aliases:
            control_id = self.control_ids.get(_normalize_control_name(alias))
            if control_id is None:
                continue
            control = V4L2Control()
            control.id = control_id
            control.value = int(value)
            try:
                _xioctl(self.fd, VIDIOC_S_CTRL, control)
            except OSError:
                continue
            self.applied_controls[name] = int(value)
            return True
        return False

    def _apply_controls(self) -> None:
        ordered_names = [
            "exposure_dynamic_framerate",
            "auto_exposure",
            "exposure_absolute",
            "auto_wb",
            "wb_temperature",
        ]
        ordered_names.extend(name for name in self.controls if name not in ordered_names)
        for name in ordered_names:
            if name not in self.controls:
                continue
            if not self._set_control_by_alias(name, int(self.controls[name])):
                self.missing_controls.append(name)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

# Camera configuration
CAMERA_DEVICES = {
    "left": "/dev/video_left",
    "overhead": "/dev/video_overhead",
    "right": "/dev/video_right",
    "torso": "/dev/video_torso",
    "test": "/dev/video_test",
    "extra": "/dev/video_extra",
    "extra_overhead": "/dev/video_extra_overhead",
}

# Default frame size
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 50
FRAME_CHANNELS = 3

# Shared memory layout (must match camera_utils/capture.py)
SHM_PREFIX = "flash_aloha_cam_"
SHM_METADATA_SIZE = 16  # 8 bytes frame_number (Q) + 8 bytes timestamp (d)

# RoboLogger endpoint layout (shared by DistributedRecorder and camera clients)
CAMERA_ENDPOINT_OFFSETS = {
    "left": 0,
    "overhead": 1,
    "right": 2,
    "torso": 3,
    "test": 4,
    "extra": 5,
    "extra_overhead": 6,
}

shutdown_requested = False


def signal_handler(sig, frame):
    """Handle Ctrl+C by setting shutdown flag."""
    del sig, frame
    global shutdown_requested
    print("\n[Camera Server] Shutdown requested...")
    shutdown_requested = True


def get_camera_info(cap, name):
    """Get camera properties."""
    return {
        "name": name,
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }


def open_v4l2_camera(config):
    """Open a native V4L2 mmap camera using the resolved saved mode/settings."""
    cap = V4L2MmapCamera(
        device_path=config.device_path,
        width=config.width,
        height=config.height,
        fps=config.fps,
        controls=config.settings,
    )
    cap.open()
    return cap


def _cleanup_robologger_logger(logger):
    """Best-effort cleanup to release RMQ sockets promptly."""
    if logger is None:
        return
    try:
        import atexit

        try:
            atexit.unregister(logger.on_exit)
        except Exception:
            pass

        if hasattr(logger, "rmq_server") and logger.rmq_server is not None:
            if hasattr(logger.rmq_server, "close"):
                logger.rmq_server.close()
            if hasattr(logger.rmq_server, "socket") and logger.rmq_server.socket is not None:
                logger.rmq_server.socket.close(linger=0)
            logger.rmq_server = None
    except Exception as e:
        print(f"[RoboLogger] Logger cleanup warning: {e}")


def _build_camera_video_logger(cam_name, width, height, fps, endpoint_base, codec):
    """Create one VideoLogger for one camera stream."""
    try:
        from robologger.loggers.video_logger import VideoLogger
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Robologger was requested but is not installed. Install the local "
            "Robologger package or start this camera with --no-robologger."
        ) from exc

    if cam_name not in CAMERA_ENDPOINT_OFFSETS:
        raise ValueError(f"Unknown camera name for endpoint mapping: {cam_name}")

    endpoint_port = endpoint_base + CAMERA_ENDPOINT_OFFSETS[cam_name]
    logger_name = f"video_{cam_name}"
    return VideoLogger(
        name=logger_name,
        endpoint=f"tcp://0.0.0.0:{endpoint_port}",
        attr={
            "camera_configs": {
                "main": {
                    "width": width,
                    "height": height,
                    "fps": fps,
                    "type": "rgb",
                }
            }
        },
        codec=codec,
    ), endpoint_port


class CameraRecordingWriter:
    """Bounded camera writer whose thread exclusively owns logger lifecycle.

    Capturing threads call :meth:`submit_frame` and never wait for FFmpeg.  The
    oldest queued frame is discarded on overflow; its original timestamp still
    makes the remaining stream causally alignable.
    """

    def __init__(
        self,
        camera_logger,
        camera_name,
        fps,
        *,
        queue_size=128,
        warmup_frames=5,
        heartbeat_interval=0.25,
        persistent_failure_count=3,
        encoder_settle_seconds=1.5,
        startup_retry_limit=1,
        encoder_resource_fallback_codec="libx264",
        profile_log_timing=False,
    ):
        self._logger_factory = camera_logger if callable(camera_logger) else None
        self.camera_logger = None if self._logger_factory is not None else camera_logger
        self.camera_name = str(camera_name)
        self.fps = float(fps)
        self.queue_size = max(2, int(queue_size))
        self.warmup_frames = max(1, int(warmup_frames))
        self.heartbeat_interval = max(0.05, float(heartbeat_interval))
        self.persistent_failure_count = max(1, int(persistent_failure_count))
        self.encoder_settle_seconds = max(0.0, float(encoder_settle_seconds))
        self.startup_retry_limit = max(0, int(startup_retry_limit))
        self.encoder_resource_fallback_codec = str(
            encoder_resource_fallback_codec or ""
        ).strip()
        self.profile_log_timing = bool(profile_log_timing)
        self._frames = deque()
        self._condition = threading.Condition()
        self._shutdown = False
        self._accepting = False
        self._state = "idle"
        self._session_id = None
        self._episode_path = None
        self._command_id = None
        self._pending_pause = None
        self._pending_resume = None
        self._pending_stop = None
        self._pending_start = None
        self._pending_restart = None
        self._start_ack_command = None
        self._successful_stable_writes = 0
        self._consecutive_failures = 0
        self._first_write_failure = None
        self._encoder_started_monotonic = None
        self._startup_attempt = 0
        self._ffmpeg_report_path = None
        initial_logger = None if self._logger_factory is not None else self.camera_logger
        self._configured_encoder_codec = str(
            getattr(initial_logger, "codec", "") or ""
        )
        self._encoder_fallback_reason = None
        self._timing_file = None
        self._source_color_order = "legacy_passthrough"
        self._continuity_id = 0
        self._queue_high_water = 0
        self._max_write_delay = 0.0
        self._accepted_frames = 0
        self.logged_frames = 0
        self.dropped_frames = 0
        self.last_timestamp = None
        self.fatal_error = None
        self._last_status_time = 0.0
        self._seen_commands = set()
        self._thread = threading.Thread(
            target=self._run,
            name=f"{self.camera_name}-recording-writer",
            daemon=False,
        )
        self._initialized = threading.Event()

    @property
    def accepting(self):
        with self._condition:
            return self._accepting

    @property
    def queue_depth(self):
        with self._condition:
            return len(self._frames)

    @property
    def state(self):
        with self._condition:
            return self._state

    def start(self):
        self._thread.start()

    def wait_initialized(self, timeout=5.0):
        if not self._initialized.wait(timeout=float(timeout)):
            raise RuntimeError(f"Camera writer {self.camera_name} initialization timed out")
        if self.fatal_error is not None:
            raise RuntimeError(self.fatal_error)

    def close(self, timeout=15.0):
        with self._condition:
            self._shutdown = True
            self._accepting = False
            self._condition.notify_all()
        self._thread.join(timeout=float(timeout))
        if self._thread.is_alive():
            raise RuntimeError(f"Camera writer {self.camera_name} did not stop")

    def submit_frame(self, frame, timestamp, frame_number, frame_ready_time, *, raw_timestamp=None,
                     buffer_flags=None, driver_sequence=None):
        with self._condition:
            if not self._accepting or self.fatal_error is not None:
                return False
            if len(self._frames) >= self.queue_size:
                self._frames.popleft()
                self.dropped_frames += 1
            self._frames.append(
                (frame.copy(), float(raw_timestamp if raw_timestamp is not None else timestamp),
                 int(frame_number), float(frame_ready_time),
                 float(raw_timestamp) if raw_timestamp is not None else float(timestamp),
                 self._continuity_id, buffer_flags, driver_sequence)
            )
            self._queue_high_water = max(self._queue_high_water, len(self._frames))
            self._accepted_frames += 1
            self._condition.notify()
            return True

    def submit_command(self, command):
        """Test/local command injection; RMQ commands use the same path."""
        with self._condition:
            self._handle_command_locked(dict(command))
            self._condition.notify_all()

    def status(self, *, command_id=None):
        with self._condition:
            encoder = self._encoder_metadata()
            return {
                "name": f"video_{self.camera_name}",
                "camera": self.camera_name,
                "state": self._state,
                "session_id": self._session_id,
                "command_id": self._command_id if command_id is None else command_id,
                "episode_path": self._episode_path,
                "ffmpeg_healthy": self._ffmpeg_healthy_locked(),
                "queue_depth": len(self._frames),
                "queue_capacity": self.queue_size,
                "timing_schema_version": 2,
                "timestamp_source": "v4l2_monotonic",
                "continuity_id": self._continuity_id,
                "queue_high_water": self._queue_high_water,
                "queued_seconds": len(self._frames) / self.fps,
                "oldest_queued_age_seconds": max(0., time.monotonic() - self._frames[0][3]) if self._frames else 0.,
                "max_write_delay_seconds": self._max_write_delay,
                "logged_frames": self.logged_frames,
                "accepted_frames": self._accepted_frames,
                "last_timestamp": self.last_timestamp,
                "dropped_frames": self.dropped_frames,
                "video_encoder": encoder,
                "fatal_error": self.fatal_error,
                "heartbeat_monotonic": time.monotonic(),
            }

    def _ffmpeg_healthy_locked(self):
        if self.camera_logger is None:
            return False
        if self._state in {"idle", "stopped"}:
            return True
        processes = getattr(self.camera_logger, "ffmpeg_processes", {})
        return bool(processes) and all(process.poll() is None for process in processes.values())

    def _publish_status(self, *, command_id=None):
        payload = self.status(command_id=command_id)
        try:
            from robotmq import serialize

            self.camera_logger.rmq_server.put_data("info", serialize(payload))
        except Exception as exc:
            # Losing the status channel is a recorder failure: the coordinator
            # can no longer safely release the robot from a lifecycle barrier.
            with self._condition:
                self.fatal_error = self.fatal_error or f"status publish failed: {exc}"
                self._state = "error"

    def _poll_commands(self):
        try:
            from robotmq import deserialize

            raw, _ = self.camera_logger.rmq_server.pop_data("command", n=0)
            for item in raw:
                command = deserialize(item)
                if isinstance(command, dict):
                    self.submit_command(command)
        except Exception as exc:
            with self._condition:
                self.fatal_error = self.fatal_error or f"command polling failed: {exc}"
                self._state = "error"

    def _handle_command_locked(self, command):
        kind = command.get("type")
        session_id = str(command.get("session_id", "legacy"))
        command_id = str(command.get("command_id", f"legacy-{time.monotonic_ns()}"))
        command["session_id"] = session_id
        command["command_id"] = command_id
        command_key = (session_id, command_id)
        if command_key in self._seen_commands:
            return
        self._seen_commands.add(command_key)
        if kind == "start":
            if self._state not in {"idle", "stopped"}:
                if (
                    session_id != self._session_id
                    and self._state
                    in {"warming", "ready", "pausing", "paused", "resuming"}
                ):
                    # A coordinator can disappear while the persistent camera
                    # server still owns an open (often paused) recording.  A
                    # start for a new session is an explicit ownership handoff:
                    # drain and close the stale recording before opening the
                    # new one.  Retrying start within the same session remains
                    # an error unless it has the same command id and was caught
                    # by _seen_commands above.
                    self._accepting = False
                    self._state = "restarting"
                    self._pending_pause = None
                    self._pending_resume = None
                    self._pending_stop = None
                    self._pending_start = None
                    self._pending_restart = dict(command)
                    self._start_ack_command = dict(command)
                    self._session_id = session_id
                    self._command_id = command_id
                    self._episode_path = str(command["episode_dir"])
                    self._startup_attempt = 0
                    self.warmup_frames = max(
                        1, int(command.get("warmup_frames", self.warmup_frames))
                    )
                    return
                self.fatal_error = f"start received while state={self._state}"
                self._state = "error"
                return
            self._session_id = session_id
            self._command_id = command_id
            self._episode_path = str(command["episode_dir"])
            self._startup_attempt = 0
            self.warmup_frames = max(
                1, int(command.get("warmup_frames", self.warmup_frames))
            )
            self._state = "starting"
            self._pending_start = dict(command)
            self._start_ack_command = dict(command)
        elif session_id != self._session_id:
            # Stale commands from a prior session are explicitly ignored.
            return
        elif kind == "pause":
            self._accepting = False
            self._state = "pausing"
            self._pending_pause = dict(command)
        elif kind == "resume":
            if self._state != "paused":
                self.fatal_error = f"resume received while state={self._state}"
                self._state = "error"
                return
            self._command_id = command_id
            self._accepting = True
            self._continuity_id += 1
            self._state = "resuming"
            self._pending_resume = dict(command)
        elif kind == "stop":
            self._accepting = False
            self._state = "stopping"
            self._pending_stop = dict(command)
        else:
            self.fatal_error = f"unknown recording command: {kind!r}"
            self._state = "error"

    def _upgrade_timestamp_dtype(self):
        group = getattr(self.camera_logger, "zarr_group", None)
        if group is None:
            raise RuntimeError("VideoLogger did not initialize Zarr storage")
        data_lists = getattr(self.camera_logger, "data_lists", None)
        for camera_key in self.camera_logger.attr["camera_configs"]:
            name = f"{camera_key}_timestamps"
            # Newer robologger releases buffer timestamp samples in data_lists
            # and materialize the float64 Zarr dataset in stop_recording().
            # Older releases create a float32 dataset up front.  Support both
            # storage contracts instead of requiring a dataset that deliberately
            # does not exist yet in the buffered implementation.
            if isinstance(data_lists, dict) and name in data_lists:
                continue
            if name not in group:
                raise RuntimeError(
                    f"VideoLogger initialized neither {name!r} nor its in-memory buffer"
                )
            if np.dtype(group[name].dtype) == np.dtype(np.float64):
                continue
            del group[name]
            group.create_dataset(name, shape=(0,), chunks=(1000,), dtype=np.float64)

    def _timestamp_count(self, camera_key="main"):
        name = f"{camera_key}_timestamps"
        data_lists = getattr(self.camera_logger, "data_lists", None)
        if isinstance(data_lists, dict) and name in data_lists:
            return len(data_lists[name])
        group = getattr(self.camera_logger, "zarr_group", None)
        if group is not None and name in group:
            return int(group[name].shape[0])
        raise RuntimeError(
            f"VideoLogger has no timestamp storage for camera {camera_key!r}"
        )

    def _start_logger(self, command):
        self._source_color_order = command.get("source_color_order", "legacy_passthrough")
        if self._source_color_order not in {"legacy_passthrough", "rgb", "bgr"}:
            raise ValueError(f"Unknown recording source color order: {self._source_color_order}")
        self._startup_attempt += 1
        report_path = (
            Path(command["episode_dir"])
            / f"video_{self.camera_name}_ffmpeg_attempt{self._startup_attempt}.log"
        )
        previous_ffreport = os.environ.get("FFREPORT")
        os.environ["FFREPORT"] = f"file={report_path}:level=32"
        try:
            self.camera_logger.start_recording(command["episode_dir"])
        finally:
            if previous_ffreport is None:
                os.environ.pop("FFREPORT", None)
            else:
                os.environ["FFREPORT"] = previous_ffreport
        self._upgrade_timestamp_dtype()
        self._persist_encoder_metadata()
        if self._timing_file is not None:
            self._timing_file.close()
        self._timing_file = (Path(command["episode_dir"]) / f"video_{self.camera_name}.timing.jsonl").open("w")
        with self._condition:
            self._frames.clear()
            self._accepting = True
            self._state = "warming"
            self._successful_stable_writes = 0
            self._consecutive_failures = 0
            self._first_write_failure = None
            self._encoder_started_monotonic = time.monotonic()
            self._ffmpeg_report_path = str(report_path)
            self._continuity_id = 0
            self._queue_high_water = 0
            self._max_write_delay = 0.0
            self._accepted_frames = 0
            self.logged_frames = 0
            self.dropped_frames = 0
            self.last_timestamp = None

    def _write_frame(self, item):
        frame, timestamp, source_frame_number, frame_ready_time, raw_timestamp, continuity_id, buffer_flags, driver_sequence = item
        started = time.monotonic()
        # Camera readout is OpenCV BGR; the installed VideoLogger feeds rgb24
        # to FFmpeg. Opt in per collection session, leaving legacy clients alone.
        encoded_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if self._source_color_order == "bgr" else frame
        self.camera_logger.log_frame(
            camera_name="main", timestamp=timestamp, frame=encoded_frame
        )
        finished = time.monotonic()
        elapsed = finished - started
        # One timing row per video frame. Encoding delays never change capture time.
        self._timing_file.write(json.dumps(dict(
            video_frame=self.logged_frames, frame_number=source_frame_number,
            timestamp=timestamp, capture_timestamp=raw_timestamp,
            frame_ready_timestamp=frame_ready_time, write_started_timestamp=started,
            write_finished_timestamp=finished, continuity_id=continuity_id,
            timestamp_source='v4l2_monotonic', buffer_flags=buffer_flags,
            driver_sequence=driver_sequence)) + "\n")
        if (self.logged_frames + 1) % 100 == 0:
            self._timing_file.flush()
        with self._condition:
            self._max_write_delay = max(self._max_write_delay, started - frame_ready_time)
            self.logged_frames += 1
            self.last_timestamp = timestamp
            self._consecutive_failures = 0
            self._first_write_failure = None
            if elapsed <= 2.0 / self.fps:
                self._successful_stable_writes += 1
            else:
                self._successful_stable_writes = 0
            if self.profile_log_timing:
                print(
                    f"[LogTime] {self.camera_name} frame={source_frame_number} "
                    f"ready_to_log={(started-frame_ready_time)*1000:.3f} ms "
                    f"log_frame={elapsed*1000:.3f} ms"
                )

    def _finish_stop(self, command):
        self._validate_recording_counts()
        self.camera_logger.stop_recording()
        if self._timing_file is not None:
            self._timing_file.close()
            self._timing_file = None
        with self._condition:
            self._state = "stopped"
            self._command_id = command["command_id"]
            self._pending_stop = None
        self._publish_status(command_id=command["command_id"])
        capture_gc.collect_idle()  # the only collection point while this server streams

    def _validate_recording_counts(self):
        timestamp_count = self._timestamp_count("main")
        if timestamp_count != self.logged_frames:
            raise RuntimeError(
                f"timestamp/frame count mismatch: {timestamp_count} != {self.logged_frames}"
            )
        if self.logged_frames + self.dropped_frames != self._accepted_frames:
            raise RuntimeError(
                "accepted frame accounting mismatch: "
                f"logged={self.logged_frames}, dropped={self.dropped_frames}, "
                f"accepted={self._accepted_frames}"
            )

    def _restart_logger(self, command):
        # Preserve the stale episode as a valid, closed camera recording.  Its
        # abandoned coordinator may later mark the overall manifest incomplete.
        self._validate_recording_counts()
        self.camera_logger.stop_recording()
        self._start_logger(command)

    def _retry_warming_logger(self, command, failure):
        fallback_codec = self._nvenc_resource_fallback_codec()
        retry_detail = (
            f"falling back to {fallback_codec}"
            if fallback_codec is not None
            else "retrying the configured encoder"
        )
        print(
            f"[RoboLogger] [{self.camera_name}] FFmpeg failed during cold start "
            f"attempt {self._startup_attempt}; {retry_detail}: {failure}. "
            f"Diagnostics: {self._ffmpeg_report_path}"
        )
        self.camera_logger.stop_recording()
        if fallback_codec is not None:
            self.camera_logger.codec = fallback_codec
            self._encoder_fallback_reason = "nvenc_resource_failure"
        self._start_logger(command)

    def _encoder_metadata(self):
        effective_codec = str(getattr(self.camera_logger, "codec", "") or "")
        configured_codec = self._configured_encoder_codec or effective_codec
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        return {
            "source_color_order": self._source_color_order,
            "configured_codec": configured_codec,
            "effective_codec": effective_codec,
            "backend": "nvenc" if "nvenc" in effective_codec.lower() else "software",
            "fallback_used": bool(
                configured_codec and effective_codec and effective_codec != configured_codec
            ),
            "fallback_reason": self._encoder_fallback_reason,
            "cuda_visible_devices": cuda_visible_devices,
        }

    def _persist_encoder_metadata(self):
        group = getattr(self.camera_logger, "zarr_group", None)
        if group is None:
            return
        group.attrs['timestamp_source'] = 'v4l2_monotonic'
        group.attrs['timestamp_clock'] = 'CLOCK_MONOTONIC seconds'
        metadata = self._encoder_metadata()
        for key, value in metadata.items():
            if value is not None:
                group.attrs[f"video_encoder_{key}"] = value

    def _nvenc_resource_fallback_codec(self):
        """Return a CPU codec only for a confirmed NVENC resource failure."""
        fallback = self.encoder_resource_fallback_codec
        codec = str(getattr(self.camera_logger, "codec", ""))
        if not fallback or "nvenc" not in codec.lower() or fallback == codec:
            return None
        report_path = self._ffmpeg_report_path
        if not report_path:
            return None
        try:
            report = Path(report_path).read_text(errors="replace").lower()
        except OSError:
            return None
        resource_markers = (
            "out of memory",
            "cannot allocate memory",
            "cuda_error_out_of_memory",
            "no capable devices found",
            "cannot load libcuda",
        )
        return fallback if any(marker in report for marker in resource_markers) else None

    def _run(self):
        try:
            if self._logger_factory is not None:
                self.camera_logger = self._logger_factory()
            if not self._configured_encoder_codec:
                self._configured_encoder_codec = str(
                    getattr(self.camera_logger, "codec", "") or ""
                )
            self._initialized.set()
            while True:
                self._poll_commands()
                with self._condition:
                    if self._shutdown and not self._frames:
                        break
                    start_command = self._pending_start
                    self._pending_start = None
                    frame_item = self._frames.popleft() if self._frames else None
                    restart_command = (
                        self._pending_restart
                        if self._pending_restart and frame_item is None
                        else None
                    )
                    if restart_command is not None:
                        self._pending_restart = None
                    pause_command = (
                        self._pending_pause if self._pending_pause and not self._frames else None
                    )
                    if pause_command is not None:
                        self._pending_pause = None
                    stop_command = (
                        self._pending_stop if self._pending_stop and not self._frames else None
                    )
                    if (
                        frame_item is None
                        and start_command is None
                        and restart_command is None
                        and pause_command is None
                        and stop_command is None
                    ):
                        self._condition.wait(timeout=0.01)

                if start_command is not None:
                    self._start_logger(start_command)
                if restart_command is not None:
                    self._restart_logger(restart_command)
                if frame_item is not None:
                    retry_command = None
                    try:
                        self._write_frame(frame_item)
                    except Exception as exc:
                        with self._condition:
                            self._consecutive_failures += 1
                            self.dropped_frames += 1
                            if self._first_write_failure is None:
                                self._first_write_failure = str(exc)
                            if (
                                self._state == "warming"
                                and self._startup_attempt <= self.startup_retry_limit
                                and self._start_ack_command is not None
                            ):
                                self._accepting = False
                                self._state = "retrying"
                                retry_command = dict(self._start_ack_command)
                            elif (
                                "FFmpeg process" in str(exc)
                                or self._consecutive_failures
                                >= self.persistent_failure_count
                            ):
                                self.fatal_error = (
                                    "FFmpeg writer failure: "
                                    f"{self._first_write_failure}. Diagnostics: "
                                    f"{self._ffmpeg_report_path}"
                                )
                                self._state = "error"
                                self._accepting = False
                    if retry_command is not None:
                        self._retry_warming_logger(
                            retry_command, self._first_write_failure
                        )
                        continue
                    with self._condition:
                        if (
                            self._state == "warming"
                            and self._successful_stable_writes >= self.warmup_frames
                            and time.monotonic() - self._encoder_started_monotonic
                            >= self.encoder_settle_seconds
                            and len(self._frames) < self.queue_size / 2
                            and self._ffmpeg_healthy_locked()
                        ):
                            self._state = "ready"
                            command = self._start_ack_command
                            self._command_id = command["command_id"]
                            self._start_ack_command = None
                            publish_command_id = command["command_id"]
                        elif self._state == "resuming":
                            command = self._pending_resume
                            self._pending_resume = None
                            self._state = "ready"
                            self._command_id = command["command_id"]
                            publish_command_id = command["command_id"]
                        else:
                            publish_command_id = None
                    if publish_command_id is not None:
                        self._publish_status(command_id=publish_command_id)
                if pause_command is not None:
                    with self._condition:
                        self._state = "paused"
                        self._command_id = pause_command["command_id"]
                    self._publish_status(command_id=pause_command["command_id"])
                if stop_command is not None:
                    self._finish_stop(stop_command)

                now = time.monotonic()
                if now - self._last_status_time >= self.heartbeat_interval:
                    self._publish_status()
                    self._last_status_time = now
        except Exception as exc:
            with self._condition:
                self.fatal_error = self.fatal_error or str(exc)
                self._state = "error"
                self._accepting = False
            self._initialized.set()
            self._publish_status()
        finally:
            if self.camera_logger is not None and getattr(self.camera_logger, "_is_recording", False):
                try:
                    self.camera_logger.stop_recording()
                except Exception as exc:
                    with self._condition:
                        self.fatal_error = self.fatal_error or f"shutdown flush failed: {exc}"
            if self._timing_file is not None:
                self._timing_file.close()
                self._timing_file = None
            _cleanup_robologger_logger(self.camera_logger)


class SharedFrameBuffer:
    """Shared memory writer for one camera stream."""

    def __init__(self, camera_name, width, height):
        self.camera_name = camera_name
        self.width = int(width)
        self.height = int(height)
        self.frame_size = self.width * self.height * FRAME_CHANNELS
        self.total_size = SHM_METADATA_SIZE + self.frame_size
        self.shm_name = f"{SHM_PREFIX}{camera_name}"
        self.shm = None
        self._create()

    def _create(self):
        try:
            old = shared_memory.SharedMemory(name=self.shm_name)
            old.close()
            old.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

        self.shm = shared_memory.SharedMemory(
            name=self.shm_name, create=True, size=self.total_size
        )
        publish_shared_clock_contract(self.shm)

    def write_frame(self, frame, frame_number, timestamp):
        if self.shm is None:
            return

        frame_bytes = frame.tobytes()
        if len(frame_bytes) != self.frame_size:
            raise ValueError(
                f"Frame byte size mismatch for {self.camera_name}: "
                f"expected {self.frame_size}, got {len(frame_bytes)}"
            )

        start = SHM_METADATA_SIZE
        end = SHM_METADATA_SIZE + self.frame_size

        # Mark the header busy BEFORE changing pixels. Readers verify the same
        # non-busy header before/after copying so the timestamp belongs to the
        # image, even when they overlap publication.
        self.shm.buf[:SHM_METADATA_SIZE] = struct.pack('Qd', int(frame_number) | (1 << 63), float('nan'))
        self.shm.buf[start:end] = frame_bytes
        self.shm.buf[:SHM_METADATA_SIZE] = struct.pack("Qd", int(frame_number), float(timestamp))

    def cleanup(self):
        if self.shm is not None:
            try:
                self.shm.close()
            except Exception:
                pass
            try:
                self.shm.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass
            self.shm = None


def run_camera_server(
    camera_name,
    target_fps=None,
    width=None,
    height=None,
    timestamp_fps=None,
    use_robologger=True,
    robologger_endpoint_base=55551,
    robologger_codec="h264_nvenc",
    profile_log_timing=False,
    smooth_timestamps=False,
    timestamp_smoothing_window=30,
    write_shared_memory=False,
    log_queue_size=128,
    max_log_drain_per_loop=8,
    recording_warmup_frames=5,
    read_fail_threshold=15,
    reopen_backoff_ms=200,
):
    """Run a single-camera capture loop in this process."""
    global shutdown_requested

    if smooth_timestamps:
        raise ValueError('Synthetic camera timestamps are no longer supported; use driver capture timestamps')

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if camera_name not in CAMERA_DEVICES:
        print(f"[ERROR] Unknown camera name: {camera_name}")
        print(f"Valid options: {sorted(CAMERA_DEVICES.keys())}")
        return 1

    device_path = CAMERA_DEVICES[camera_name]
    camera_config = resolve_camera_config(
        name=camera_name,
        device_path=device_path,
        width_override=width,
        height_override=height,
        fps_override=target_fps,
        fallback_width=DEFAULT_WIDTH,
        fallback_height=DEFAULT_HEIGHT,
        fallback_fps=DEFAULT_FPS,
    )

    target_fps = float(camera_config.fps)
    if target_fps <= 0:
        raise ValueError(f"target_fps must be > 0, got {target_fps}")

    if timestamp_fps is None:
        timestamp_fps = target_fps
    timestamp_fps = float(timestamp_fps)
    if timestamp_fps <= 0:
        raise ValueError(f"timestamp_fps must be > 0, got {timestamp_fps}")

    try:
        cap = open_v4l2_camera(camera_config)
    except Exception as e:
        print(f"[ERROR] Failed to initialize camera {camera_name!r} ({device_path}): {e}")
        return 1

    info = get_camera_info(cap, camera_name)
    reported_fps = float(info["fps"]) if info["fps"] > 0 else 0.0
    capture_fps_label = reported_fps if reported_fps > 0 else target_fps
    log_queue_size = max(1, int(log_queue_size))
    del max_log_drain_per_loop  # Kept as a deprecated CLI compatibility option.
    read_fail_threshold = max(1, int(read_fail_threshold))
    reopen_backoff_s = max(0.0, float(reopen_backoff_ms) / 1000.0)

    print("=" * 60)
    print("  Flash Aloha Camera Server (Single Camera)")
    print("=" * 60)
    print(f"Camera: {camera_name} ({device_path})")
    if camera_config.loaded_from_params:
        print(f"Params: loaded saved mode/controls from {camera_config.params_path}")
    else:
        print(
            f"Params: no saved record for {camera_name}; using fallback "
            f"{DEFAULT_WIDTH}x{DEFAULT_HEIGHT}@{DEFAULT_FPS}"
        )
    print(
        f"Mode: {info['width']}x{info['height']} @ requested {target_fps:.3f} FPS "
        f"(driver reports {capture_fps_label:.3f} FPS)"
    )
    print("Pixel format: MJPG via native V4L2 mmap, decoded to BGR")
    if getattr(cap, "applied_controls", None):
        applied = ", ".join(
            f"{name}={value}" for name, value in sorted(cap.applied_controls.items())
        )
        print(f"Controls: {applied}")
    if getattr(cap, "missing_controls", None):
        missing = ", ".join(sorted(set(cap.missing_controls)))
        print(f"[WARN] Controls not available on {camera_name}: {missing}")
    print('Timestamps: V4L2 driver CLOCK_MONOTONIC, unchanged by recording or encoding')
    if profile_log_timing:
        print("[Profiling] Enabled [TsDiff] and [LogTime] prints")
    print(
        f"[Logging] background-writer queue={log_queue_size}, "
        f"read_fail_threshold={read_fail_threshold}, reopen_backoff={reopen_backoff_ms}ms"
    )

    shared_buffer = None
    if write_shared_memory:
        try:
            shared_buffer = SharedFrameBuffer(
                camera_name=camera_name, width=info["width"], height=info["height"]
            )
            print(
                f"[SharedMem] Writing frames to {shared_buffer.shm_name} "
                f"({info['width']}x{info['height']})"
            )
        except Exception as e:
            cap.release()
            print(f"[SharedMem] Failed to initialize shared memory for {camera_name}: {e}")
            return 1

    recording_writer = None
    if use_robologger:
        try:
            endpoint_port = (
                robologger_endpoint_base + CAMERA_ENDPOINT_OFFSETS[camera_name]
            )
            logger_factory = lambda: _build_camera_video_logger(
                cam_name=camera_name,
                width=int(info["width"]),
                height=int(info["height"]),
                fps=timestamp_fps,
                endpoint_base=robologger_endpoint_base,
                codec=robologger_codec,
            )[0]
            recording_writer = CameraRecordingWriter(
                logger_factory,
                camera_name,
                timestamp_fps,
                queue_size=log_queue_size,
                warmup_frames=recording_warmup_frames,
                profile_log_timing=profile_log_timing,
            )
            recording_writer.start()
            recording_writer.wait_initialized()
            print(
                f"[RoboLogger] {camera_name} logger ready at "
                f"tcp://0.0.0.0:{endpoint_port} (name=video_{camera_name})"
            )
        except Exception as e:
            if recording_writer is not None:
                try:
                    recording_writer.close()
                except Exception:
                    pass
            cap.release()
            print(f"[RoboLogger] Failed to initialize logger for {camera_name}: {e}")
            return 1

    frame_number = 0
    frames_since_stats = 0
    logged_frame_count = 0
    actual_fps = 0.0
    actual_log_fps = 0.0
    last_stats_time = time.time()
    last_camera_timestamp = None
    last_raw_camera_timestamp = None
    raw_dt_window = deque(maxlen=max(1, int(timestamp_smoothing_window)))
    consecutive_read_failures = 0
    total_read_failures = 0
    last_reopen_wait_log = 0.0
    last_frame_ready_time = None

    print("[Camera Server] Running... Press Ctrl+C to stop")
    print("=" * 60)

    capture_gc.pause_for_stream()
    try:
        while not shutdown_requested:
            state_update_start = time.monotonic()
            is_recording = bool(recording_writer and recording_writer.accepting)
            writer_state = recording_writer.state if recording_writer else "idle"
            state_update_ms = (time.monotonic() - state_update_start) * 1000.0

            ret = False
            frame = None
            if cap is None:
                now_mono = time.monotonic()
                if now_mono - last_reopen_wait_log >= 2.0:
                    print(f"[ReinitWait] {camera_name} waiting to reopen camera...")
                    last_reopen_wait_log = now_mono
                if reopen_backoff_s > 0:
                    time.sleep(reopen_backoff_s)
                try:
                    cap = open_v4l2_camera(camera_config)
                except Exception as e:
                    print(f"[ReinitFail] {camera_name} reopen failed: {e}")
                    cap = None
                if cap is not None:
                    reopened_info = get_camera_info(cap, camera_name)
                    reopened_fps = (
                        float(reopened_info["fps"])
                        if reopened_info["fps"] > 0
                        else capture_fps_label
                    )
                    print(
                        f"[ReinitOK] {camera_name} reopened "
                        f"({reopened_info['width']}x{reopened_info['height']} @ "
                        f"driver reports {reopened_fps:.3f} FPS)"
                    )
                    consecutive_read_failures = 0
                    last_raw_camera_timestamp = None
                    last_camera_timestamp = None
                    raw_dt_window.clear()

            raw_camera_timestamp = 0.0
            read_start = time.monotonic()
            if cap is not None:
                ret, frame, raw_camera_timestamp = cap.read()
            read_end = time.monotonic()

            if ret:
                capture_timestamp = read_end
                consecutive_read_failures = 0
                read_elapsed_ms = (read_end - read_start) * 1000.0
                frame_ready_gap_ms = (
                    None
                    if last_frame_ready_time is None
                    else (capture_timestamp - last_frame_ready_time) * 1000.0
                )
                raw_frame_age_ms = (
                    (capture_timestamp - raw_camera_timestamp) * 1000.0
                )
                last_frame_ready_time = capture_timestamp

                raw_dt_ms = None
                if last_raw_camera_timestamp is not None:
                    raw_dt = raw_camera_timestamp - last_raw_camera_timestamp
                    raw_dt_ms = raw_dt * 1000.0
                    if raw_dt > 0:
                        raw_dt_window.append(raw_dt)
                last_raw_camera_timestamp = raw_camera_timestamp

                camera_timestamp = raw_camera_timestamp

                if profile_log_timing:
                    ready_gap_label = (
                        "nan"
                        if frame_ready_gap_ms is None
                        else f"{frame_ready_gap_ms:.3f}"
                    )
                    print(
                        f"[CaptureTime] {camera_name} frame={frame_number} "
                        f"ready_mono={capture_timestamp:.6f} "
                        f"ready_gap={ready_gap_label} ms "
                        f"read={read_elapsed_ms:.3f} ms "
                        f"raw_age={raw_frame_age_ms:.3f} ms "
                        f"state_update={state_update_ms:.3f} ms "
                        f"recording={int(is_recording)}"
                    )
                    if last_camera_timestamp is None:
                        print(
                            f"[TsDiff] {camera_name} frame={frame_number} "
                            f"ts={camera_timestamp:.6f} dt=nan"
                        )
                    else:
                        dt_ms = (camera_timestamp - last_camera_timestamp) * 1000.0
                        if raw_dt_ms is None:
                            print(
                                f"[TsDiff] {camera_name} frame={frame_number} "
                                f"ts={camera_timestamp:.6f} dt={dt_ms:.3f} ms"
                            )
                        else:
                            rolling_raw_ms = (
                                (sum(raw_dt_window) / len(raw_dt_window)) * 1000.0
                                if raw_dt_window else float("nan")
                            )
                            print(
                                f"[TsDiff] {camera_name} frame={frame_number} "
                                f"ts={camera_timestamp:.6f} dt={dt_ms:.3f} ms "
                                f"(raw={raw_dt_ms:.3f} ms, raw_avg={rolling_raw_ms:.3f} ms)"
                            )
                    last_camera_timestamp = camera_timestamp

                if shared_buffer is not None:
                    try:
                        shared_buffer.write_frame(frame, frame_number, camera_timestamp)
                    except Exception as e:
                        print(f"[SharedMem] [{camera_name}] write_frame failed: {e}")
                        shared_buffer.cleanup()
                        shared_buffer = None

                if recording_writer is not None:
                    recording_writer.submit_frame(
                        frame,
                        camera_timestamp,
                        frame_number,
                        capture_timestamp,
                        raw_timestamp=raw_camera_timestamp,
                        buffer_flags=cap.last_buffer_flags,
                        driver_sequence=cap.last_sequence,
                    )

                frame_number += 1
                frames_since_stats += 1
            elif cap is not None:
                consecutive_read_failures += 1
                total_read_failures += 1

                if (
                    consecutive_read_failures == 1
                    or consecutive_read_failures % read_fail_threshold == 0
                ):
                    print(
                        f"[ReadFail] {camera_name} consecutive={consecutive_read_failures} "
                        f"total={total_read_failures}"
                    )

                if consecutive_read_failures >= read_fail_threshold:
                    print(
                        f"[ReinitStart] {camera_name} reopening after "
                        f"{consecutive_read_failures} consecutive read failures"
                    )
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = None

            now = time.time()
            if now - last_stats_time >= 5.0:
                window = now - last_stats_time
                actual_fps = frames_since_stats / window
                writer_logged = recording_writer.logged_frames if recording_writer else 0
                writer_dropped = recording_writer.dropped_frames if recording_writer else 0
                writer_depth = recording_writer.queue_depth if recording_writer else 0
                newly_logged = max(0, writer_logged - logged_frame_count)
                actual_log_fps = newly_logged / window
                logged_frame_count = writer_logged
                print(
                    f"[Stats] {camera_name}: {actual_fps:.1f}/{target_fps:.1f} FPS "
                    f"(frame {frame_number}), log {actual_log_fps:.1f} FPS "
                    f"(logged {logged_frame_count}), queue {writer_depth}/{log_queue_size}, "
                    f"drop {writer_dropped}, read_fail {total_read_failures}"
                )
                last_stats_time = now
                frames_since_stats = 0

    finally:
        print("\n[Camera Server] Shutting down...")
        if recording_writer is not None:
            try:
                recording_writer.close()
            except Exception as e:
                print(f"[RoboLogger] writer shutdown failed for {camera_name}: {e}")
        if shared_buffer is not None:
            shared_buffer.cleanup()
        if cap is not None:
            cap.release()
        print(f"[OK] Stopped {camera_name}")
        print("[Camera Server] Shutdown complete")

    return 0


def _resolve_camera_arg(camera, cameras):
    """Resolve camera name from args while keeping --cameras backward compatibility."""
    if camera is not None:
        return camera

    parsed = [c.strip() for c in cameras.split(",") if c.strip()]
    if len(parsed) == 1:
        return parsed[0]

    raise ValueError(
        "This server now supports one camera per process. "
        "Use --camera <name> and start multiple processes (one per camera)."
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Single-camera capture server for Flash Aloha. "
            "Run one process per camera."
        )
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Camera to capture (left,overhead,right,torso,test)",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default="left",
        help=(
            "Deprecated. Accepts only one camera value. "
            "For multiple cameras, run multiple camera_server.py processes."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help=(
            "Target capture FPS requested from the camera. "
            "Defaults to calibration/camera_params.npz, then 50."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Capture width override. Defaults to saved camera params, then 1280.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Capture height override. Defaults to saved camera params, then 720.",
    )
    parser.add_argument(
        "--timestamp-fps",
        type=float,
        default=None,
        help=(
            "Nominal FPS for video encoding/playback metadata only. Does not alter capture timestamps. Defaults to --fps."
        ),
    )
    parser.add_argument(
        "--use-robologger",
        action="store_true",
        default=True,
        help="Enable robologger VideoLogger endpoint (default: enabled)",
    )
    parser.add_argument(
        "--no-robologger",
        action="store_false",
        dest="use_robologger",
        help="Disable robologger VideoLogger endpoint",
    )
    parser.add_argument(
        "--robologger-endpoint-base",
        type=int,
        default=55551,
        help=(
            "Base TCP port for camera logger endpoints. "
            "left uses base+0, overhead base+1, right base+2, torso base+3"
        ),
    )
    parser.add_argument(
        "--robologger-codec",
        type=str,
        default="h264_nvenc",
        help="Video codec used by robologger VideoLogger (default: h264_nvenc)",
    )
    parser.add_argument(
        "--profile-log-timing",
        action="store_true",
        help="Print per-frame [TsDiff] and per-log [LogTime] timing",
    )
    parser.add_argument(
        "--no-smooth-timestamps",
        action="store_false",
        dest="smooth_timestamps",
        default=False,
        help=(
            "Compatibility option: timestamps always use the V4L2 driver capture clock."
        ),
    )
    parser.add_argument(
        "--timestamp-smoothing-window",
        type=int,
        default=30,
        help=(
            "Rolling window size (frames) for raw timestamp diagnostics "
            "when profiling is enabled (default: 30)."
        ),
    )
    parser.add_argument(
        "--write-shared-memory",
        action="store_true",
        help=(
            "Also publish latest frame to shared memory "
            "(flash_aloha_cam_<camera>) for inference readers."
        ),
    )
    parser.add_argument(
        "--log-queue-size",
        type=int,
        default=128,
        help=(
            "Max queued frames waiting for robologger encoding (default: 128). "
            "If full, oldest frame is dropped."
        ),
    )
    parser.add_argument(
        "--max-log-drain-per-loop",
        type=int,
        default=8,
        help=(
            "Deprecated compatibility option; encoding now runs on a dedicated thread."
        ),
    )
    parser.add_argument(
        "--recording-warmup-frames",
        type=int,
        default=5,
        help="Consecutive stable FFmpeg writes required before ready (default: 5).",
    )
    parser.add_argument(
        "--read-fail-threshold",
        type=int,
        default=15,
        help=(
            "Consecutive cap.read() failures before auto-reopen (default: 15)."
        ),
    )
    parser.add_argument(
        "--cpu-latency-us",
        type=int,
        default=20,
        help=(
            "Hold this CPU wake-up latency limit while streaming so deep idle states "
            "cannot drop USB frames (default: 20; negative disables)."
        ),
    )
    parser.add_argument(
        "--reopen-backoff-ms",
        type=int,
        default=200,
        help=(
            "Delay before camera reopen attempts in milliseconds (default: 200)."
        ),
    )
    args = parser.parse_args()

    try:
        camera_name = _resolve_camera_arg(args.camera, args.cameras)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return 2

    from camera_utils.cpu_latency import cpu_latency_limit

    with cpu_latency_limit(args.cpu_latency_us):
        return run_camera_server(
            camera_name=camera_name,
            target_fps=args.fps,
            width=args.width,
            height=args.height,
            timestamp_fps=args.timestamp_fps,
            use_robologger=args.use_robologger,
            robologger_endpoint_base=args.robologger_endpoint_base,
            robologger_codec=args.robologger_codec,
            profile_log_timing=args.profile_log_timing,
            smooth_timestamps=args.smooth_timestamps,
            timestamp_smoothing_window=args.timestamp_smoothing_window,
            write_shared_memory=args.write_shared_memory,
            log_queue_size=args.log_queue_size,
            max_log_drain_per_loop=args.max_log_drain_per_loop,
            recording_warmup_frames=args.recording_warmup_frames,
            read_fail_threshold=args.read_fail_threshold,
            reopen_backoff_ms=args.reopen_backoff_ms,
        )


if __name__ == "__main__":
    raise SystemExit(main())

