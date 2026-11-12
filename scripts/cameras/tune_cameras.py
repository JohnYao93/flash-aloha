#!/usr/bin/env python3
"""
Interactive camera control tuner.

Controls:
  Up/Down     Select parameter
  Left/Right  Decrease/increase selected parameter
  [ / ]       Switch camera when multiple cameras are selected
  v           Run V4L2 mmap FPS test on the active camera
  w           Briefly auto-WB, then lock manual WB
  1           Set auto_exposure=1 and auto_wb=1 on the active camera
  c           Copy active camera settings to all selected cameras
  y           Save NPZ and exit
  n or q      Exit without saving
"""

# Support direct execution as well as package imports.
import sys as _path_sys
from pathlib import Path as _Path
_ACT_ROOT = _Path(__file__).resolve().parents[2] / "src"
if str(_ACT_ROOT) not in _path_sys.path:
    _path_sys.path.insert(0, str(_ACT_ROOT))
_SCRIPT_ROOT = _ACT_ROOT.parent / "scripts"
if str(_SCRIPT_ROOT) not in _path_sys.path:
    _path_sys.path.append(str(_SCRIPT_ROOT))

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, replace

import cv2
import numpy as np


REPO_ROOT = str(_ACT_ROOT.parent)
ACT_SRC = os.path.join(REPO_ROOT, "src")
if ACT_SRC not in sys.path:
    sys.path.insert(0, ACT_SRC)

from camera_utils.camera_server import V4L2MmapCamera


DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 60
# TORSO_DEFAULT_WIDTH = 1280
# TORSO_DEFAULT_HEIGHT = 800
# TORSO_DEFAULT_FPS = 100
# TORSO_CAMERA_DEVICE = "/dev/video_torso"

CAMERA_DEVICES = {
    "left": "/dev/video_left",
    "overhead": "/dev/video_overhead",
    "right": "/dev/video_right",
    # "torso": "/dev/video_torso",
    # "video_torso": "/dev/video_torso",
    "test": "/dev/video_test",
}


KEY_LEFT = {81, 2424832, 65361}
KEY_UP = {82, 2490368, 65362}
KEY_RIGHT = {83, 2555904, 65363}
KEY_DOWN = {84, 2621440, 65364}


@dataclass(frozen=True)
class ControlSpec:
    name: str
    prop: int
    minimum: int
    maximum: int
    step: int
    values: tuple[int, ...] | None = None
    v4l2_name: str | None = None
    v4l2_aliases: tuple[str, ...] = ()


CONTROL_SPECS = (
    ControlSpec("brightness", cv2.CAP_PROP_BRIGHTNESS, -64, 64, 1, v4l2_name="brightness"),
    ControlSpec("contrast", cv2.CAP_PROP_CONTRAST, 0, 64, 1, v4l2_name="contrast"),
    ControlSpec("saturation", cv2.CAP_PROP_SATURATION, 0, 128, 1, v4l2_name="saturation"),
    ControlSpec("hue", cv2.CAP_PROP_HUE, -40, 40, 1, v4l2_name="hue"),
    ControlSpec(
        "auto_wb",
        cv2.CAP_PROP_AUTO_WB,
        0,
        1,
        1,
        (0, 1),
        v4l2_name="white_balance_automatic",
    ),
    ControlSpec("gamma", cv2.CAP_PROP_GAMMA, 72, 500, 1, v4l2_name="gamma"),
    ControlSpec("gain", cv2.CAP_PROP_GAIN, 0, 255, 1, v4l2_name="gain"),
    ControlSpec(
        "power_line_frequency",
        -1,
        0,
        2,
        1,
        (0, 1, 2),
        v4l2_name="power_line_frequency",
    ),
    ControlSpec(
        "wb_temperature",
        cv2.CAP_PROP_WB_TEMPERATURE,
        2800,
        6500,
        20,
        v4l2_name="white_balance_temperature",
    ),
    ControlSpec(
        "sharpness",
        getattr(cv2, "CAP_PROP_SHARPNESS", -1),
        0,
        6,
        1,
        v4l2_name="sharpness",
    ),
    ControlSpec(
        "backlight_compensation",
        getattr(cv2, "CAP_PROP_BACKLIGHT", -1),
        0,
        2,
        1,
        v4l2_name="backlight_compensation",
    ),
    # OpenCV V4L2 commonly reports 1=manual and 3=auto/aperture priority.
    ControlSpec(
        "auto_exposure",
        cv2.CAP_PROP_AUTO_EXPOSURE,
        1,
        3,
        2,
        (1, 3),
        v4l2_name="auto_exposure",
    ),
    ControlSpec(
        "exposure_absolute",
        cv2.CAP_PROP_EXPOSURE,
        1,
        10000,
        5,
        v4l2_name="exposure_time_absolute",
    ),
    ControlSpec(
        "exposure_dynamic_framerate",
        -1,
        0,
        1,
        1,
        (0, 1),
        v4l2_name="exposure_dynamic_framerate",
    ),
    ControlSpec(
        "autofocus",
        cv2.CAP_PROP_AUTOFOCUS,
        0,
        1,
        1,
        (0, 1),
        v4l2_name="focus_auto",
        v4l2_aliases=("focus_automatic",),
    ),
    ControlSpec(
        "focus",
        cv2.CAP_PROP_FOCUS,
        0,
        255,
        1,
        v4l2_name="focus_absolute",
    ),
)
SPEC_BY_NAME = {spec.name: spec for spec in CONTROL_SPECS}
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
CONTROL_APPLY_ORDER = (
    "exposure_dynamic_framerate",
    "auto_exposure",
    "exposure_absolute",
    "auto_wb",
    "wb_temperature",
)


def order_control_specs_for_apply(control_specs: tuple[ControlSpec, ...]) -> tuple[ControlSpec, ...]:
    specs_by_name = {spec.name: spec for spec in control_specs}
    ordered = [specs_by_name[name] for name in CONTROL_APPLY_ORDER if name in specs_by_name]
    ordered.extend(spec for spec in control_specs if spec.name not in CONTROL_APPLY_ORDER)
    return tuple(ordered)


def resolve_camera(camera: str) -> tuple[str, str]:
    if camera in CAMERA_DEVICES:
        return camera, CAMERA_DEVICES[camera]
    if camera.startswith("/dev/"):
        basename = os.path.basename(camera)
        # if basename == "video_torso":
        #     return "torso", "/dev/video_torso"
        return basename.replace("video_", "").replace("video", "cam"), camera
    if camera.isdigit():
        return f"cam{camera}", f"/dev/video{camera}"
    if camera.startswith("video_"):
        return camera.replace("video_", ""), f"/dev/{camera}"
    return camera, f"/dev/video_{camera}"


# def is_torso_device(device_path: str) -> bool:
#     if device_path in {TORSO_CAMERA_DEVICE, "/dev/video_torso"}:
#         return True
#     try:
#         return os.path.realpath(device_path) == os.path.realpath(TORSO_CAMERA_DEVICE)
#     except OSError:
#         return device_path in {TORSO_CAMERA_DEVICE, "/dev/video_torso"}


def resolve_capture_mode(
    device_path: str,
    width: int | None,
    height: int | None,
    fps: int | None,
) -> tuple[int, int, int]:
    del device_path
    return (
        DEFAULT_WIDTH if width is None else width,
        DEFAULT_HEIGHT if height is None else height,
        DEFAULT_FPS if fps is None else fps,
    )


def run_v4l2_ctl(device_path: str, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device_path, *args],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()
    except FileNotFoundError:
        return None


def set_v4l2_control(device_path: str, name: str, value: int) -> None:
    run_v4l2_ctl(device_path, f"--set-ctrl={name}={value}")


def get_v4l2_control_specs(device_path: str) -> dict[str, dict[str, int]]:
    output = run_v4l2_ctl(device_path, "--list-ctrls")
    if not output:
        return {}

    controls = {}
    pattern = re.compile(r"^\s*(?P<name>\S+)\s+0x[0-9a-fA-F]+.*?: (?P<body>.*)$")
    for line in output.splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        body = match.group("body")
        control = {}
        for key, field in (
            ("minimum", "min"),
            ("maximum", "max"),
            ("step", "step"),
            ("default", "default"),
        ):
            field_match = re.search(rf"\b{field}=(-?\d+)", body)
            if field_match is not None:
                control[key] = int(field_match.group(1))
        if "step" in control:
            control["step"] = max(1, abs(control["step"]))
        controls[match.group("name")] = control
    return controls


def get_available_control_specs(device_path: str) -> tuple[ControlSpec, ...]:
    v4l2_controls = get_v4l2_control_specs(device_path)
    if not v4l2_controls:
        return CONTROL_SPECS

    specs = []
    for spec in CONTROL_SPECS:
        v4l2_name = spec.v4l2_name
        if v4l2_name not in v4l2_controls:
            v4l2_name = next(
                (alias for alias in spec.v4l2_aliases if alias in v4l2_controls),
                None,
            )
        if v4l2_name is None:
            continue
        control = v4l2_controls[v4l2_name]
        minimum = control.get("minimum", spec.minimum)
        maximum = control.get("maximum", spec.maximum)
        step = max(spec.step, control.get("step", spec.step))
        values = spec.values
        if values is not None:
            values = tuple(value for value in values if minimum <= value <= maximum)
            if not values:
                values = None
        specs.append(
            replace(
                spec,
                minimum=minimum,
                maximum=maximum,
                step=step,
                values=values,
                v4l2_name=v4l2_name,
            )
        )
    return tuple(specs)


def configure_v4l2_stream(device_path: str, width: int, height: int, fps: int) -> None:
    run_v4l2_ctl(
        device_path,
        f"--set-fmt-video=width={width},height={height},pixelformat=MJPG",
        f"--set-parm={fps}",
    )


def apply_v4l2_settings(device_path: str, settings: dict[str, int]) -> None:
    available_specs = get_available_control_specs(device_path)
    for spec in order_control_specs_for_apply(available_specs):
        if spec.name in settings and spec.v4l2_name is not None:
            set_v4l2_control(device_path, spec.v4l2_name, settings[spec.name])


def run_v4l2_stream_test(
    device_path: str,
    width: int,
    height: int,
    fps: int,
    frame_count: int,
    settings: dict[str, int],
    timeout_s: float | None,
) -> None:
    estimated_s = frame_count / max(float(fps), 1.0)
    timeout_s = timeout_s if timeout_s is not None else max(5.0, estimated_s + 3.0)
    print(
        f"[V4L2] Testing {device_path} at {width}x{height}@{fps} "
        f"for {frame_count} frames (timeout {timeout_s:.1f}s)"
    )
    apply_v4l2_settings(device_path, settings)
    try:
        result = subprocess.run(
            [
                "v4l2-ctl",
                "-d",
                device_path,
                f"--set-fmt-video=width={width},height={height},pixelformat=MJPG",
                f"--set-parm={fps}",
                "--stream-mmap",
                f"--stream-count={frame_count}",
                "--stream-to=/dev/null",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=timeout_s,
        )
    except FileNotFoundError:
        print("[V4L2] v4l2-ctl not found")
        return
    except subprocess.TimeoutExpired as exc:
        output = "\n".join(
            part.decode(errors="replace") if isinstance(part, bytes) else part
            for part in (exc.stdout, exc.stderr)
            if part
        ).strip()
        if output:
            print(output)
        print(
            f"[V4L2] Stream test timed out after {timeout_s:.1f}s. "
            "The camera did not deliver the requested frames fast enough."
        )
        return

    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if output:
        print(output)
    if result.returncode != 0:
        print(f"[V4L2] Stream test failed with exit code {result.returncode}")


def get_v4l2_control(device_path: str, name: str) -> int | None:
    output = run_v4l2_ctl(device_path, f"--get-ctrl={name}")
    if not output or ":" not in output:
        return None
    raw_value = output.rsplit(":", maxsplit=1)[-1].strip()
    try:
        return int(raw_value)
    except ValueError:
        return None


def open_camera(device_path: str, width: int, height: int, fps: int) -> cv2.VideoCapture | None:
    if not os.path.exists(device_path):
        print(f"[WARN] {device_path} does not exist")
        return None

    configure_v4l2_stream(device_path, width, height, fps)

    cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERR] Failed to open {device_path}")
        return None

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    set_v4l2_control(device_path, "exposure_dynamic_framerate", 0)
    return cap


def open_v4l2_preview_camera(
    device_path: str,
    width: int,
    height: int,
    fps: int,
    settings: dict[str, int],
) -> V4L2MmapCamera | None:
    if not os.path.exists(device_path):
        print(f"[WARN] {device_path} does not exist")
        return None

    try:
        cap = V4L2MmapCamera(
            device_path=device_path,
            width=width,
            height=height,
            fps=fps,
            controls=settings,
        )
        cap.open()
        return cap
    except Exception as exc:
        print(f"[ERR] Failed to open {device_path} with native V4L2 mmap: {exc}")
        return None


def read_camera_frame(cap) -> tuple[bool, np.ndarray | None]:
    result = cap.read()
    if isinstance(result, tuple) and len(result) == 3:
        ret, frame, _timestamp = result
        return ret, frame
    return result


def set_control(
    cap: cv2.VideoCapture,
    device_path: str,
    spec: ControlSpec,
    value: int,
) -> int:
    if spec.values is not None:
        value = min(spec.values, key=lambda option: abs(option - value))
    value = int(np.clip(value, spec.minimum, spec.maximum))
    if spec.prop >= 0 and hasattr(cap, "set"):
        cap.set(spec.prop, value)
    if spec.v4l2_name is not None:
        set_v4l2_control(device_path, spec.v4l2_name, value)
        value = get_v4l2_control(device_path, spec.v4l2_name) or value
    return value


def get_control(
    cap: cv2.VideoCapture,
    device_path: str,
    spec: ControlSpec,
    fallback: int,
) -> int:
    if spec.v4l2_name is not None:
        value = get_v4l2_control(device_path, spec.v4l2_name)
        if value is not None:
            return value
    if spec.prop >= 0 and hasattr(cap, "get"):
        value = cap.get(spec.prop)
        if value >= 0:
            return int(round(value))
    return fallback


def apply_settings(
    cap: cv2.VideoCapture,
    device_path: str,
    settings: dict[str, int],
    control_specs: tuple[ControlSpec, ...],
) -> None:
    for spec in order_control_specs_for_apply(control_specs):
        if spec.name in settings:
            settings[spec.name] = set_control(cap, device_path, spec, settings[spec.name])


def read_settings(
    cap: cv2.VideoCapture,
    device_path: str,
    settings: dict[str, int],
    control_specs: tuple[ControlSpec, ...],
) -> dict[str, int]:
    result = dict(settings)
    for spec in control_specs:
        result[spec.name] = get_control(cap, device_path, spec, settings[spec.name])
    return result


def next_value(spec: ControlSpec, current: int, direction: int) -> int:
    if spec.values is not None:
        values = list(spec.values)
        nearest_idx = min(range(len(values)), key=lambda i: abs(values[i] - current))
        return values[(nearest_idx + direction) % len(values)]
    return int(np.clip(current + direction * spec.step, spec.minimum, spec.maximum))


def settle_and_lock_wb(
    cap: cv2.VideoCapture,
    device_path: str,
    settings: dict[str, int],
    control_specs: tuple[ControlSpec, ...],
    settle_frames: int = 30,
) -> dict[str, int]:
    specs_by_name = {spec.name: spec for spec in control_specs}
    if "auto_wb" not in specs_by_name:
        print("[WB] auto_wb is not available on this camera")
        return settings

    settings["auto_wb"] = set_control(cap, device_path, specs_by_name["auto_wb"], 1)
    frames = 0
    for _ in range(max(1, int(settle_frames))):
        ret, _ = read_camera_frame(cap)
        if ret:
            frames += 1
    settings = read_settings(cap, device_path, settings, control_specs)
    locked_wb = settings.get("wb_temperature")
    settings["auto_wb"] = set_control(cap, device_path, specs_by_name["auto_wb"], 0)
    if locked_wb is not None and "wb_temperature" in specs_by_name:
        settings["wb_temperature"] = set_control(
            cap,
            device_path,
            specs_by_name["wb_temperature"],
            locked_wb,
        )
    settings = read_settings(cap, device_path, settings, control_specs)
    print(
        f"[WB] Settled for {frames} frame(s), "
        f"locked auto_wb=0 wb_temperature={settings.get('wb_temperature', 'unknown')}"
    )
    return settings


def control_label(spec: ControlSpec) -> str:
    if spec.v4l2_name is not None and spec.v4l2_name != spec.name:
        return f"{spec.name} [{spec.v4l2_name}]"
    return spec.name


def format_control_line(spec: ControlSpec, settings: dict[str, int]) -> str:
    value = settings[spec.name]
    text = f"{control_label(spec)}: {value}"
    if spec.name == "auto_wb" and value == 1 and "wb_temperature" in settings:
        text += f" (wb {settings['wb_temperature']})"
    return text


def draw_overlay(
    frame: np.ndarray,
    camera_name: str,
    device_path: str,
    camera_index: int,
    camera_count: int,
    settings: dict[str, int],
    selected_idx: int,
    display_fps: float,
    control_specs: tuple[ControlSpec, ...],
    backend: str = "v4l2"
) -> np.ndarray:
    overlay = frame.copy()
    panel_height = max(190, 118 + len(control_specs) * 28)
    cv2.rectangle(overlay, (0, 0), (680, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    lines = [
        f"Camera {camera_index + 1}/{camera_count}: {camera_name} ({device_path})",
        f"FPS: {display_fps:.1f}  Backend: {backend}",
        "Up/Down select  Left/Right tune  c copy all  w lock WB  y save",
    ]
    for row, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (14, 28 + row * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    for idx, spec in enumerate(control_specs):
        prefix = ">" if idx == selected_idx else " "
        value = settings[spec.name]
        text = f"{prefix} {format_control_line(spec, settings)}"
        color = (0, 255, 255) if idx == selected_idx else (210, 210, 210)
        cv2.putText(
            frame,
            text,
            (14, 118 + idx * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return frame


def scalar_or_index(values: np.ndarray, idx: int) -> int | None:
    values = np.asarray(values)
    if values.ndim == 0:
        return int(values)
    if idx < values.shape[0]:
        return int(values[idx])
    return None


def camera_match_keys(name: str, device: str) -> set[str]:
    keys = {str(name), str(device)}
    basename = os.path.basename(str(device))
    if basename:
        keys.add(basename)
        keys.add(basename.replace("video_", ""))
    if str(name).startswith("video_"):
        keys.add(str(name).replace("video_", ""))
    else:
        keys.add(f"video_{name}")
    return {key for key in keys if key}


def load_npz_records(path: str) -> list[dict]:
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
                for spec in CONTROL_SPECS:
                    value = None
                    if spec.name in params:
                        value = scalar_or_index(params[spec.name], idx)
                    elif values is not None and spec.name in param_names and idx < values.shape[0]:
                        value = int(values[idx, param_names.index(spec.name)])
                    if value is not None:
                        settings[spec.name] = value

                width = scalar_or_index(params["widths"], idx) if "widths" in params else None
                height = scalar_or_index(params["heights"], idx) if "heights" in params else None
                fps = scalar_or_index(params["fps_values"], idx) if "fps_values" in params else None
                if width is None and "width" in params:
                    width = scalar_or_index(params["width"], idx)
                if height is None and "height" in params:
                    height = scalar_or_index(params["height"], idx)
                if fps is None and "fps" in params:
                    fps = scalar_or_index(params["fps"], idx)

                records.append(
                    {
                        "name": name,
                        "device": device,
                        "settings": settings,
                        "mode": (width, height, fps) if None not in (width, height, fps) else None,
                    }
                )
    except Exception as exc:
        print(f"[WARN] Failed to load existing camera params from {path}: {exc}")
    return records


def find_record(records: list[dict], name: str, device: str) -> dict | None:
    keys = camera_match_keys(name, device)
    for record in records:
        if keys & camera_match_keys(record["name"], record["device"]):
            return record
    return None


def complete_settings(settings: dict[str, int]) -> dict[str, int]:
    result = dict(DEFAULT_CONTROL_SETTINGS)
    result.update({key: int(value) for key, value in settings.items()})
    return result


def default_mode_for(name: str, device: str) -> tuple[int, int, int]:
    del name
    return resolve_capture_mode(device, None, None, None)


def build_save_records(
    cameras: list[tuple[str, str]],
    all_settings: dict[str, dict[str, int]],
    camera_modes: dict[str, tuple[int, int, int]],
    existing_records: list[dict],
) -> list[dict]:
    selected = {name: (device, all_settings[name], camera_modes[name]) for name, device in cameras}
    records = []
    updated = set()

    for record in existing_records:
        match_name = None
        for name, (device, _, _) in selected.items():
            if camera_match_keys(name, device) & camera_match_keys(record["name"], record["device"]):
                match_name = name
                break

        if match_name is None:
            mode = record.get("mode") or default_mode_for(record["name"], record["device"])
            records.append(
                {
                    "name": record["name"],
                    "device": record["device"],
                    "settings": complete_settings(record.get("settings", {})),
                    "mode": mode,
                }
            )
            continue

        device, settings, mode = selected[match_name]
        records.append(
            {
                "name": match_name,
                "device": device,
                "settings": complete_settings(settings),
                "mode": mode,
            }
        )
        updated.add(match_name)

    for name, (device, settings, mode) in selected.items():
        if name in updated:
            continue
        records.append(
            {
                "name": name,
                "device": device,
                "settings": complete_settings(settings),
                "mode": mode,
            }
        )

    return records


def save_npz(
    output_path: str,
    cameras: list[tuple[str, str]],
    all_settings: dict[str, dict[str, int]],
    camera_modes: dict[str, tuple[int, int, int]],
    existing_records: list[dict] | None = None,
) -> None:
    records = build_save_records(cameras, all_settings, camera_modes, existing_records or [])
    camera_names = np.array([record["name"] for record in records])
    devices = np.array([record["device"] for record in records])
    param_names = np.array([spec.name for spec in CONTROL_SPECS])
    widths = np.array([record["mode"][0] for record in records], dtype=np.int32)
    heights = np.array([record["mode"][1] for record in records], dtype=np.int32)
    fps_values = np.array([record["mode"][2] for record in records], dtype=np.int32)
    values = np.array(
        [[record["settings"][spec.name] for spec in CONTROL_SPECS] for record in records],
        dtype=np.int32,
    )

    save_kwargs = {
        "camera_names": camera_names,
        "devices": devices,
        "param_names": param_names,
        "values": values,
        "width": widths[0] if np.all(widths == widths[0]) else widths,
        "height": heights[0] if np.all(heights == heights[0]) else heights,
        "fps": fps_values[0] if np.all(fps_values == fps_values[0]) else fps_values,
        "widths": widths,
        "heights": heights,
        "fps_values": fps_values,
    }
    for col, spec in enumerate(CONTROL_SPECS):
        save_kwargs[spec.name] = values[:, col]

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez(output_path, **save_kwargs)
    print(f"[INFO] Saved camera parameters to {output_path}")


def parse_cameras(raw: str) -> list[tuple[str, str]]:
    cameras = []
    seen = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        name, device = resolve_camera(item)
        if name in seen:
            continue
        cameras.append((name, device))
        seen.add(name)
    if not cameras:
        raise ValueError("No cameras selected")
    return cameras


def tune(args: argparse.Namespace) -> int:
    cameras = parse_cameras(args.cameras)
    existing_records = load_npz_records(args.output)
    camera_modes = {}
    all_settings = {}
    cli_setting_overrides = {
        "brightness": args.brightness,
        "contrast": args.contrast,
        "saturation": args.saturation,
        "hue": args.hue,
        "auto_wb": args.auto_wb,
        "gamma": args.gamma,
        "gain": args.gain,
        "power_line_frequency": args.power_line_frequency,
        "wb_temperature": args.wb_temperature,
        "sharpness": args.sharpness,
        "backlight_compensation": args.backlight_compensation,
        "auto_exposure": args.auto_exposure,
        "exposure_absolute": args.exposure_absolute,
        "exposure_dynamic_framerate": args.exposure_dynamic_framerate,
        "autofocus": args.autofocus,
        "focus": args.focus,
    }
    cli_setting_overrides = {
        key: value for key, value in cli_setting_overrides.items() if value is not None
    }

    for name, device in cameras:
        existing_record = find_record(existing_records, name, device)
        default_width, default_height, default_fps = default_mode_for(name, device)
        saved_mode = existing_record.get("mode") if existing_record else None
        saved_width, saved_height, saved_fps = saved_mode or (None, None, None)
        camera_modes[name] = (
            args.width if args.width is not None else (saved_width or default_width),
            args.height if args.height is not None else (saved_height or default_height),
            args.fps if args.fps is not None else (saved_fps or default_fps),
        )

        settings = complete_settings(existing_record.get("settings", {}) if existing_record else {})
        settings.update({key: int(value) for key, value in cli_setting_overrides.items()})
        all_settings[name] = settings
        if existing_record is not None:
            print(f"[INFO] Loaded saved params for {name} from {args.output}")

    print("Controls:")
    print("  Up/Down     Select parameter")
    print("  Left/Right  Decrease/increase selected parameter")
    print("  [ / ]       Switch camera")
    print("  v           Run V4L2 mmap FPS test")
    print(f"  backend     {args.backend}")
    print("  w           Briefly auto-WB, then lock manual WB")
    print("  1           Set auto_exposure=1 and auto_wb=1")
    print("  c           Copy active camera settings to all selected cameras")
    print("  y           Save NPZ and exit")
    print("  n or q      Exit without saving")

    camera_idx = 0
    selected_idx = 0
    active_specs = CONTROL_SPECS
    cap = None
    window_name = "Camera Tuner"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    fps_start = time.time()
    display_fps = 0.0
    last_settings_refresh = 0.0

    def load_active_camera():
        nonlocal active_specs, selected_idx, last_settings_refresh

        name, device = cameras[camera_idx]
        width, height, fps = camera_modes[name]
        active_specs = get_available_control_specs(device)
        if not active_specs:
            print(f"[WARN] No tunable controls found for {name}; using full control list")
            active_specs = CONTROL_SPECS
        selected_idx %= len(active_specs)

        if args.backend == "v4l2":
            active_cap = open_v4l2_preview_camera(device, width, height, fps, all_settings[name])
        else:
            active_cap = open_camera(device, width, height, fps)
        if active_cap is None:
            return None

        apply_settings(active_cap, device, all_settings[name], active_specs)
        control_names = ", ".join(control_label(spec) for spec in active_specs)
        last_settings_refresh = 0.0
        print(f"[INFO] Tuning {name} ({device}) at {width}x{height}@{fps} via {args.backend}")
        print(f"[INFO] Available controls: {control_names}")
        return active_cap

    cap = load_active_camera()
    if cap is None:
        return 1

    try:
        while True:
            name, device = cameras[camera_idx]
            ret, frame = read_camera_frame(cap)
            if not ret:
                print(f"[WARN] Failed to read frame from {name}")
                time.sleep(0.05)
                key = cv2.waitKeyEx(1)
            else:
                frame_count += 1
                elapsed = time.time() - fps_start
                if elapsed >= 1.0:
                    display_fps = frame_count / elapsed
                    frame_count = 0
                    fps_start = time.time()

                now = time.time()
                if (
                    args.settings_refresh_interval == 0
                    or now - last_settings_refresh >= args.settings_refresh_interval
                ):
                    all_settings[name] = read_settings(cap, device, all_settings[name], active_specs)
                    last_settings_refresh = now
                display = frame
                if args.scale != 1.0:
                    display = cv2.resize(
                        frame,
                        None,
                        fx=args.scale,
                        fy=args.scale,
                        interpolation=cv2.INTER_AREA,
                    )
                display = draw_overlay(
                    display,
                    name,
                    device,
                    camera_idx,
                    len(cameras),
                    all_settings[name],
                    selected_idx,
                    display_fps,
                    active_specs,
                    args.backend,
                )
                cv2.imshow(window_name, display)
                key = cv2.waitKeyEx(1)

            if key < 0:
                continue
            if key in KEY_UP:
                selected_idx = (selected_idx - 1) % len(active_specs)
            elif key in KEY_DOWN:
                selected_idx = (selected_idx + 1) % len(active_specs)
            elif key in KEY_LEFT or key in KEY_RIGHT:
                direction = -1 if key in KEY_LEFT else 1
                spec = active_specs[selected_idx]
                current = all_settings[name][spec.name]
                all_settings[name][spec.name] = set_control(
                    cap,
                    device,
                    spec,
                    next_value(spec, current, direction),
                )
                print(f"[SET] {name}.{spec.name}={all_settings[name][spec.name]}")
            elif key == ord("1"):
                specs_by_name = {spec.name: spec for spec in active_specs}
                updated = []
                if "auto_exposure" in specs_by_name:
                    all_settings[name]["auto_exposure"] = set_control(
                        cap, device, specs_by_name["auto_exposure"], 1
                    )
                    updated.append(f"{name}.auto_exposure=1")
                if "auto_wb" in specs_by_name:
                    all_settings[name]["auto_wb"] = set_control(
                        cap,
                        device,
                        specs_by_name["auto_wb"],
                        1,
                    )
                    updated.append(f"{name}.auto_wb=1")
                if updated:
                    print(f"[SET] {' '.join(updated)}")
            elif key == ord("w"):
                all_settings[name] = settle_and_lock_wb(
                    cap,
                    device,
                    all_settings[name],
                    active_specs,
                    args.wb_settle_frames,
                )
            elif key == ord("c"):
                all_settings[name] = read_settings(cap, device, all_settings[name], active_specs)
                source_settings = dict(all_settings[name])
                for target_name, _target_device in cameras:
                    all_settings[target_name] = dict(source_settings)
                copied = ", ".join(target_name for target_name, _ in cameras)
                print(f"[COPY] Copied {name} settings to: {copied}")
            elif key == ord("v"):
                all_settings[name] = read_settings(cap, device, all_settings[name], active_specs)
                cap.release()
                width, height, fps = camera_modes[name]
                time.sleep(0.2)
                run_v4l2_stream_test(
                    device,
                    width,
                    height,
                    fps,
                    args.v4l2_test_frames,
                    all_settings[name],
                    args.v4l2_test_timeout,
                )
                cap = load_active_camera()
                if cap is None:
                    return 1
            elif key in (ord("["), ord("]")) and len(cameras) > 1:
                all_settings[name] = read_settings(cap, device, all_settings[name], active_specs)
                cap.release()
                camera_idx = (camera_idx + (-1 if key == ord("[") else 1)) % len(cameras)
                cap = load_active_camera()
                if cap is None:
                    return 1
            elif key == ord("y"):
                all_settings[name] = read_settings(cap, device, all_settings[name], active_specs)
                save_npz(args.output, cameras, all_settings, camera_modes, existing_records)
                return 0
            elif key in (ord("n"), ord("q"), 27):
                print("[INFO] Exited without saving")
                return 0
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune camera gain/WB controls live.")
    parser.add_argument(
        "--cameras",
        "-c",
        default="left,overhead,right",
        help="Comma-separated camera names or devices. Examples: left,right or /dev/video0",
    )
    parser.add_argument("--output", "-o", default="calibration/camera_params.npz")
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Capture width. Defaults to 1280 for all cameras.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Capture height. Defaults to 720.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Capture FPS. Defaults to 60.",
    )
    parser.add_argument("--scale", type=float, default=0.75)
    parser.add_argument(
        "--settings-refresh-interval",
        type=float,
        default=1.0,
        help=(
            "Seconds between V4L2 control readbacks during preview. "
            "Use 0 to read every frame, which can throttle high-FPS capture."
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("opencv", "v4l2"),
        default="v4l2",
        help=(
            "Live preview backend. Defaults to v4l2, matching the native mmap "
            "reader used by camera_server.py."
        ),
    )
    parser.add_argument(
        "--v4l2-test-frames",
        type=int,
        default=300,
        help="Number of frames for the active-camera V4L2 mmap FPS test.",
    )
    parser.add_argument(
        "--v4l2-test-timeout",
        type=float,
        default=None,
        help="Seconds before the active-camera V4L2 mmap FPS test is aborted.",
    )
    parser.add_argument("--gain", type=int, default=None)
    parser.add_argument("--exposure-absolute", type=int, default=None)
    parser.add_argument("--wb-temperature", type=int, default=None)
    parser.add_argument("--brightness", type=int, default=None)
    parser.add_argument("--contrast", type=int, default=None)
    parser.add_argument("--saturation", type=int, default=None)
    parser.add_argument("--hue", type=int, default=None)
    parser.add_argument("--gamma", type=int, default=None)
    parser.add_argument("--power-line-frequency", type=int, choices=(0, 1, 2), default=None)
    parser.add_argument("--sharpness", type=int, default=None)
    parser.add_argument("--backlight-compensation", type=int, default=None)
    parser.add_argument(
        "--exposure-dynamic-framerate",
        type=int,
        choices=(0, 1),
        default=None,
        help="Initial V4L2 exposure_dynamic_framerate value. Use 0 for stable FPS.",
    )
    parser.add_argument(
        "--wb-settle-frames",
        type=int,
        default=30,
        help="Frames to discard while briefly auto-WB settling before locking manual WB.",
    )
    parser.add_argument(
        "--auto-exposure",
        type=int,
        choices=(1, 3),
        default=None,
        help="Initial OpenCV/V4L2 auto exposure value. Commonly 1=manual, 3=auto.",
    )
    parser.add_argument("--auto-wb", type=int, choices=(0, 1), default=None)
    parser.add_argument(
        "--autofocus",
        type=int,
        choices=(0, 1),
        default=None,
        help="Initial auto-focus mode: 0=manual, 1=auto.",
    )
    parser.add_argument("--focus", type=int, default=None, help="Initial manual focus value.")
    args = parser.parse_args()
    return tune(args)


if __name__ == "__main__":
    raise SystemExit(main())
