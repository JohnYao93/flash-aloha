#!/usr/bin/env python3
"""
Livestream Camera

Simple script to livestream video from a running camera server.

Usage:
    # List available camera-server streams
    python scripts/cameras/livestream_camera.py --list

    # Start a camera server in another terminal
    uv run src/camera_utils/camera_server.py --camera overhead --write-shared-memory

    # Stream from that server
    python scripts/cameras/livestream_camera.py --camera overhead

    # Stream with an explicit expected resolution
    python scripts/cameras/livestream_camera.py --camera overhead --width 1280 --height 720

Controls:
    q - Quit
    s - Save screenshot to current directory
    f - Toggle FPS display
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
import cv2
import time
import glob
import os
import sys
from multiprocessing import resource_tracker, shared_memory
import struct

import numpy as np


ACT_SRC = str(_ACT_ROOT)
if ACT_SRC not in sys.path:
    sys.path.insert(0, ACT_SRC)

from camera_utils.camera_server import resolve_camera_config


CAMERA_DEVICES = {
    "left": "/dev/video_left",
    "overhead": "/dev/video_overhead",
    "right": "/dev/video_right",
    "torso": "/dev/video_torso",
    "test": "/dev/video_test",
}

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
FRAME_CHANNELS = 3
SHM_PREFIX = "flash_aloha_cam_"
SHM_METADATA_SIZE = 16  # 8 bytes frame_number (Q) + 8 bytes timestamp (d)


def _opencv_gui_backend():
    """Return the OpenCV GUI backend from build info, if available."""
    try:
        for line in cv2.getBuildInformation().splitlines():
            stripped = line.strip()
            if stripped.startswith("GUI:"):
                return stripped.split(":", 1)[1].strip()
    except Exception:
        return "unknown"
    return "unknown"


def _print_highgui_error(exc):
    print("[ERROR] OpenCV GUI support is unavailable in this Python environment.")
    print(f"        cv2 version: {getattr(cv2, '__version__', 'unknown')}")
    print(f"        cv2 module: {getattr(cv2, '__file__', 'unknown')}")
    print(f"        OpenCV GUI backend: {_opencv_gui_backend()}")
    print("")
    print("        This script needs cv2.namedWindow/cv2.imshow. The active venv")
    print("        is likely importing opencv-python-headless instead of opencv-python.")
    print("")
    print("        Fix the venv by reinstalling the GUI OpenCV wheel last:")
    print("          uv pip install --reinstall opencv-python==4.11.0.86")
    print("")
    print(f"        Original OpenCV error: {exc}")


class CameraServerFrameReader:
    """Reader for frames published by src/camera_utils/camera_server.py."""

    def __init__(self, camera_name, width, height):
        self.camera_name = camera_name
        self.width = int(width)
        self.height = int(height)
        self.frame_size = self.width * self.height * FRAME_CHANNELS
        self.total_size = SHM_METADATA_SIZE + self.frame_size
        self.shm_name = f"{SHM_PREFIX}{camera_name}"
        self.shm = None

    def connect(self):
        try:
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
        except FileNotFoundError:
            return False

        # This process is a viewer only; the camera server owns unlinking.
        try:
            resource_tracker.unregister(self.shm._name, "shared_memory")
        except Exception:
            pass

        if self.shm.size != self.total_size:
            actual_payload = self.shm.size - SHM_METADATA_SIZE
            self.close()
            raise ValueError(
                f"Shared memory size mismatch for {self.camera_name}: expected "
                f"{self.total_size} bytes for {self.width}x{self.height} BGR, "
                f"got {self.shm.size} bytes (payload {actual_payload}). "
                "Pass --width/--height matching the running camera server, or "
                "restart the server with matching camera params."
            )
        return True

    def read_frame(self):
        if self.shm is None and not self.connect():
            return None, -1, 0.0

        metadata = bytes(self.shm.buf[:SHM_METADATA_SIZE])
        frame_number, timestamp = struct.unpack("Qd", metadata)
        frame_bytes = bytes(self.shm.buf[SHM_METADATA_SIZE:SHM_METADATA_SIZE + self.frame_size])
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
            (self.height, self.width, FRAME_CHANNELS)
        )
        return frame.copy(), int(frame_number), float(timestamp)

    def close(self):
        if self.shm is not None:
            self.shm.close()
            self.shm = None


def list_cameras():
    """List available camera devices and camera-server shared-memory status."""
    print("=" * 60)
    print("  Camera Servers")
    print("=" * 60)

    video_devices = sorted(glob.glob('/dev/video*'))
    if video_devices:
        print("\nVideo devices found:")
        for device in video_devices:
            print(f"  {device}")

    print("\nConfigured camera server names:")
    available = []
    for name, device in CAMERA_DEVICES.items():
        config = resolve_camera_config(
            name=name,
            device_path=device,
            fallback_width=DEFAULT_WIDTH,
            fallback_height=DEFAULT_HEIGHT,
        )
        reader = CameraServerFrameReader(name, width=config.width, height=config.height)
        try:
            status = "server available" if reader.connect() else "server not available"
        except ValueError as exc:
            status = f"server available, size mismatch ({exc})"
        finally:
            reader.close()
        print(f"  {name:<8} {device:<20} expected {config.width}x{config.height} - {status}")
        if status == "server available":
            available.append(name)

    if not available:
        print("\n  No camera-server shared memory segments found")

    print("")
    return available


def _resolve_camera(camera):
    """Resolve camera input into a camera-server name and Linux /dev path."""
    if isinstance(camera, str):
        if camera.startswith("/dev/"):
            basename = os.path.basename(camera)
            if basename.startswith("video_"):
                name = basename.replace("video_", "", 1)
            else:
                name = basename
            return name, camera
        if camera.isdigit():
            return f"video{int(camera)}", f"/dev/video{int(camera)}"
        if camera in CAMERA_DEVICES:
            return camera, CAMERA_DEVICES[camera]
        if camera.startswith("video_"):
            return camera.replace("video_", "", 1), f"/dev/{camera}"
        return camera, f"/dev/video_{camera}"
    return f"video{int(camera)}", f"/dev/video{int(camera)}"


def livestream(
    camera,
    width,
    height,
    show_fps=True,
    scale=1.0,
    fps=60,
    autofocus=None,
    focus=None,
    gain=None,
):
    """
    Livestream video from a camera-server shared-memory stream.

    Args:
        camera: Camera server name, short name, or device path
        width: Expected frame width, or None to use saved camera params
        height: Expected frame height, or None to use saved camera params
        show_fps: Whether to show FPS overlay
        scale: Window scale factor (e.g., 2.0 for 2x window size)
    """
    print("=" * 60)
    print("  Camera Livestream")
    print("=" * 60)
    print(f"Camera: {camera}")
    print("Source: camera-server shared memory")
    if width is not None and height is not None:
        print(f"Expected resolution: {width}x{height}")
    else:
        print("Expected resolution: saved camera params or fallback")
    overrides = []
    if autofocus is not None:
        overrides.append(f"autofocus={autofocus}")
    if focus is not None:
        overrides.append(f"focus={focus}")
    if gain is not None:
        overrides.append(f"gain={gain}")
    override_str = ", ".join(overrides) if overrides else "none"
    print(f"Capture settings owned by camera_server.py; local overrides ignored: {override_str}, fps={fps}")
    print(f"Window scale: {scale}x")
    print("")

    camera_name, camera_device = _resolve_camera(camera)
    camera_config = resolve_camera_config(
        name=camera_name,
        device_path=camera_device,
        width_override=width,
        height_override=height,
        fallback_width=DEFAULT_WIDTH,
        fallback_height=DEFAULT_HEIGHT,
    )

    reader = CameraServerFrameReader(
        camera_name=camera_name,
        width=camera_config.width,
        height=camera_config.height,
    )
    try:
        if not reader.connect():
            print(
                f"[ERROR] Camera server shared memory not found: {reader.shm_name}\n"
                f"        Start it with: uv run src/camera_utils/camera_server.py "
                f"--camera {camera_name} --write-shared-memory"
            )
            return False
    except Exception as exc:
        print(f"[ERROR] Could not connect to camera server for {camera_name}: {exc}")
        return False

    actual_width = camera_config.width
    actual_height = camera_config.height

    print(f"[INFO] Camera server: {reader.shm_name}")
    print(f"[INFO] Device configured for server name: {camera_device}")
    if camera_config.loaded_from_params:
        print(f"[INFO] Loaded expected mode from {camera_config.params_path}")
    else:
        print(f"[INFO] No saved params for {camera_name}; using fallback expected mode")
    print(f"[INFO] Streaming at {actual_width}x{actual_height} from camera server")
    print("")
    print("Controls:")
    print("  q - Quit")
    print("  s - Save screenshot")
    print("  f - Toggle FPS display")
    print("")

    window_name = f"Livestream - {camera_name}"
    window_width = int(actual_width * scale)
    window_height = int(actual_height * scale)
    gui_initialized = False
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, window_width, window_height)
        gui_initialized = True
    except cv2.error as exc:
        reader.close()
        _print_highgui_error(exc)
        return False

    frame_count = 0
    fps_start_time = time.time()
    display_fps = 0.0
    screenshot_count = 0
    last_frame_number = -1

    try:
        while True:
            frame, frame_number, timestamp = reader.read_frame()
            if frame is None:
                print("[WARN] Failed to read frame from camera server")
                time.sleep(0.1)
                continue
            if frame_number == last_frame_number:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[INFO] Quit requested")
                    break
                elif key == ord('f'):
                    show_fps = not show_fps
                    print(f"[INFO] FPS display: {'ON' if show_fps else 'OFF'}")
                elif key == ord('s'):
                    screenshot_path = f"screenshot_{screenshot_count:04d}.png"
                    cv2.imwrite(screenshot_path, frame)
                    print(f"[INFO] Saved screenshot: {screenshot_path}")
                    screenshot_count += 1
                time.sleep(0.001)
                continue
            last_frame_number = frame_number

            frame_count += 1

            elapsed = time.time() - fps_start_time
            if elapsed >= 1.0:
                display_fps = frame_count / elapsed
                frame_count = 0
                fps_start_time = time.time()

            if scale != 1.0:
                display_frame = cv2.resize(frame, (window_width, window_height), interpolation=cv2.INTER_NEAREST)
            else:
                display_frame = frame.copy()

            if show_fps:
                fps_text = f"FPS: {display_fps:.1f}"
                cv2.putText(display_frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                res_text = f"{actual_width}x{actual_height}"
                cv2.putText(display_frame, res_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                ts_text = f"server ts: {timestamp:.3f}"
                cv2.putText(display_frame, ts_text, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Quit requested")
                break
            elif key == ord('s'):
                screenshot_path = f"screenshot_{screenshot_count:04d}.png"
                cv2.imwrite(screenshot_path, frame)
                print(f"[INFO] Saved screenshot: {screenshot_path}")
                screenshot_count += 1
            elif key == ord('f'):
                show_fps = not show_fps
                print(f"[INFO] FPS display: {'ON' if show_fps else 'OFF'}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        reader.close()
        if gui_initialized:
            cv2.destroyAllWindows()

    print("[INFO] Livestream ended")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Livestream video from a running Flash Aloha camera server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                          # List configured camera server streams
  uv run src/camera_utils/camera_server.py --camera overhead --write-shared-memory
  %(prog)s --camera overhead               # Stream from camera server shared memory
  %(prog)s --camera overhead -W 1280 -H 720 # Expected server frame size
  %(prog)s --camera overhead --scale 4     # Stream with 4x larger window

Common resolutions:
  640x480   (VGA)
  1280x720  (720p HD)
  1920x1080 (1080p Full HD)
        """
    )
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available camera-server streams')
    parser.add_argument('--camera', '-c', type=str, default='overhead',
                       help='Camera server name, short name, or device path (default: overhead)')
    parser.add_argument('--width', '-W', type=int, default=None,
                       help='Expected server frame width (default: saved camera params, then 1280)')
    parser.add_argument('--height', '-H', type=int, default=None,
                       help='Expected server frame height (default: saved camera params, then 720)')
    parser.add_argument('--fps', type=int, default=60,
                       help='Ignored; camera_server.py owns capture FPS (kept for CLI compatibility)')
    parser.add_argument('--autofocus', type=int, choices=[0, 1], default=None,
                       help='Ignored; set this on camera_server.py')
    parser.add_argument('--focus', type=int, default=None,
                       help='Ignored; set this on camera_server.py')
    parser.add_argument('--gain', type=int, default=None,
                       help='Ignored; set this on camera_server.py')
    parser.add_argument('--scale', '-s', type=float, default=3.0,
                       help='Window scale factor (default: 3.0)')
    parser.add_argument('--no-fps', action='store_true',
                       help='Hide FPS display')

    args = parser.parse_args()

    if args.list:
        list_cameras()
        return 0

    return 0 if livestream(
        args.camera,
        args.width,
        args.height,
        not args.no_fps,
        args.scale,
        fps=args.fps,
        autofocus=args.autofocus,
        focus=args.focus,
        gain=args.gain,
    ) else 1


if __name__ == '__main__':
    exit(main())
