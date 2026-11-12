"""Shared camera consumers, snapshots, and preview capture utilities."""
import os
import inspect
import struct
import threading
import time as time_module
from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory
import cv2
import numpy as np
from camera_utils.camera_timing import CameraTimingError, require_camera_timestamps, validate_shared_clock_contract
import camera_utils.robotmq_camera_cache as robotmq_camera_cache
from camera_utils.camera_server import DEFAULT_CONTROL_SETTINGS, resolve_camera_config

# A tuple assignment publishes image and capture time together. Keep the legacy
# globals for existing visualization code, but policy inputs use these packets.
captured_camera_packets = {}
_camera_packet_lock = threading.Lock()
_camera_observation_local = threading.local()


def get_camera_observation(camera_names, *, copy_images=False, validate=True):
    """Snapshot the shared consumer cache after the standard proprio read.

    RobotMQ's native receiver remains active during a Python pause. Recording
    takes only metadata; inference copies the exact referenced, pinned pixels.
    The Python compatibility backend retains its immutable packet behavior.
    """
    cache = robotmq_camera_cache.active_cache()
    if cache is not None:
        images, observation = cache.snapshot(camera_names, copy_images=copy_images, validate=validate)
        _camera_observation_local.value = observation
        return images, observation
    with _camera_packet_lock:
        cutoff = time_module.monotonic()
        packets = {name: captured_camera_packets.get(name) for name in camera_names}
    missing = [name for name, packet in packets.items() if packet is None]
    if missing:
        raise RuntimeError(f'Required timestamped camera frames are not available: {missing}')
    if any(len(packet) != 4 for packet in packets.values()):
        raise RuntimeError('Camera packets require real consumer publication timestamps; restart camera readers')
    observation = dict(snapshot_timestamp=cutoff, cameras={name: dict(
        frame_number=int(packet[1]), capture_timestamp=float(packet[2]),
        consumer_received_timestamp=float(packet[3]))
        for name, packet in packets.items()})
    stamps = {name: item['capture_timestamp'] for name, item in observation['cameras'].items()}
    if validate:
        require_camera_timestamps(stamps, now=cutoff)
    images = {name: packet[0].copy() for name, packet in packets.items()} if copy_images else {}
    # Validate at the common snapshot boundary, not after inference-only pixel
    # copying/preprocessing; collection must make the identical pairing decision.
    _camera_observation_local.value = observation
    return images, observation


def get_last_camera_observation():
    """Metadata for this caller thread's last snapshot, never a newer frame."""
    return getattr(_camera_observation_local, 'value', None)


def get_camera_bundle(camera_names, *, validate=True):
    images, observation = get_camera_observation(camera_names, copy_images=True, validate=validate)
    return images, {name: item['capture_timestamp'] for name, item in observation['cameras'].items()}

# Global variables for camera capture
captured_img_l = None
captured_img_h = None
captured_img_r = None
captured_img_t = None
captured_frame_number_l = -1
captured_frame_number_h = -1
captured_frame_number_r = -1
captured_frame_number_t = -1
captured_timestamp_l = 0.0
captured_timestamp_h = 0.0
captured_timestamp_r = 0.0
captured_timestamp_t = 0.0

# Shared memory configuration (must match camera_server.py)
SHM_PREFIX = "flash_aloha_cam_"
SHM_METADATA_SIZE = 16  # 8 bytes frame_number + 8 bytes timestamp
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
FRAME_CHANNELS = 3
_SHARED_MEMORY_SUPPORTS_TRACK = (
    "track" in inspect.signature(shared_memory.SharedMemory).parameters
)


def attach_shared_memory_non_owning(name):
    """Attach to owner-managed shared memory without scheduling an unlink at exit."""
    if _SHARED_MEMORY_SUPPORTS_TRACK:
        return shared_memory.SharedMemory(name=name, track=False)

    shm = shared_memory.SharedMemory(name=name)
    # Python < 3.13 has no public ``track=False`` argument. SharedMemory
    # registers every attachment, including readers, so explicitly remove this
    # reader from its process-local tracker. The camera server remains the sole
    # owner responsible for unlinking the block.
    resource_tracker.unregister(shm._name, "shared_memory")
    return shm


class SharedFrameReader:
    """
    Reader for shared memory frames written by camera_server.py
    """

    def __init__(self, camera_name, width=None, height=None):
        self.camera_name = camera_name
        self._explicit_dimensions = width is not None or height is not None
        if width is None or height is None:
            config = resolve_camera_config(
                name=camera_name,
                device_path=f"/dev/video_{camera_name}",
                width_override=width,
                height_override=height,
                fallback_width=DEFAULT_WIDTH,
                fallback_height=DEFAULT_HEIGHT,
            )
            width = config.width
            height = config.height
        self.width = int(width)
        self.height = int(height)
        self.frame_size = self.width * self.height * FRAME_CHANNELS
        self.total_size = self.frame_size + SHM_METADATA_SIZE
        self.shm_name = f"{SHM_PREFIX}{camera_name}"
        self.shm = None
        self.last_frame_number = -1

    def _check_size(self):
        if self.shm is None or self.shm.size == self.total_size:
            return
        actual_frame_size = self.shm.size - SHM_METADATA_SIZE
        detail = (
            f"Shared memory size mismatch for {self.camera_name}: expected "
            f"{self.total_size} bytes for {self.width}x{self.height} BGR, "
            f"got {self.shm.size} bytes"
        )
        if actual_frame_size > 0 and actual_frame_size % FRAME_CHANNELS == 0:
            detail += f" (frame payload {actual_frame_size} bytes)"
        detail += (
            ". Start camera_server.py with matching saved camera_params.npz, "
            "or pass SharedFrameReader(width=..., height=...)."
        )
        self.close()
        raise ValueError(detail)

    def connect(self):
        """Connect to shared memory. Returns True if successful."""
        try:
            self.shm = attach_shared_memory_non_owning(self.shm_name)
            self._check_size()
            try:
                validate_shared_clock_contract(self.shm)
            except Exception:
                self.close()
                raise
            return True
        except FileNotFoundError:
            return False
    
    def read_frame(self):
        """
        Read frame from shared memory.
        
        Returns:
            tuple: (frame, frame_number, timestamp) or (None, -1, 0) if failed
        """
        if self.shm is None:
            if not self.connect():
                return None, -1, 0.0
        
        try:
            for _ in range(3):
                metadata = bytes(self.shm.buf[:SHM_METADATA_SIZE])
                frame_number, timestamp = struct.unpack('Qd', metadata)
                if frame_number & (1 << 63) or not np.isfinite(timestamp) or timestamp <= 0:
                    continue
                if frame_number == self.last_frame_number:
                    return None, -1, 0.0
                frame_bytes = bytes(self.shm.buf[SHM_METADATA_SIZE:SHM_METADATA_SIZE + self.frame_size])
                if metadata != bytes(self.shm.buf[:SHM_METADATA_SIZE]):
                    continue
                frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((self.height, self.width, FRAME_CHANNELS))
                self.last_frame_number = frame_number
                with _camera_packet_lock:
                    captured_camera_packets['video_' + self.camera_name.removeprefix('video_')] = (
                        frame, frame_number, timestamp, time_module.monotonic())
                return frame.copy(), frame_number, timestamp
            return None, -1, 0.0
        except Exception as e:
            print(f"[SharedFrameReader] Error reading {self.camera_name}: {e}")
            return None, -1, 0.0
    
    def close(self):
        """Close shared memory connection."""
        if self.shm is not None:
            self.shm.close()
            self.shm = None

def get_latest_camera_frame_number(camera_name):
    """
    Return latest shared-memory frame number for a camera.

    Args:
        camera_name: Full camera name (video_left/video_overhead/video_right/video_torso)

    Returns:
        int: latest frame number, or -1 if no frame has been received yet.
    """
    cache = robotmq_camera_cache.active_cache()
    if cache is not None:
        return cache.snapshot([camera_name])[1]['cameras'][camera_name]['frame_number']
    if camera_name == 'video_left':
        return captured_frame_number_l
    if camera_name == 'video_overhead':
        return captured_frame_number_h
    if camera_name == 'video_right':
        return captured_frame_number_r
    if camera_name == 'video_torso':
        return captured_frame_number_t
    raise ValueError(
        f"Unknown camera name for frame-number lookup: {camera_name}. "
        "Expected one of: video_left, video_overhead, video_right, video_torso"
    )


def get_latest_camera_timestamp(camera_name):
    """
    Return latest camera-server timestamp from shared-memory metadata for a camera.

    Args:
        camera_name: Full camera name (video_left/video_overhead/video_right/video_torso)

    Returns:
        float: latest camera timestamp in seconds, or 0.0 if unavailable yet.
    """
    cache = robotmq_camera_cache.active_cache()
    if cache is not None:
        return cache.snapshot([camera_name])[1]['cameras'][camera_name]['capture_timestamp']
    if camera_name == 'video_left':
        return captured_timestamp_l
    if camera_name == 'video_overhead':
        return captured_timestamp_h
    if camera_name == 'video_right':
        return captured_timestamp_r
    if camera_name == 'video_torso':
        return captured_timestamp_t
    raise ValueError(
        f"Unknown camera name for timestamp lookup: {camera_name}. "
        "Expected one of: video_left, video_overhead, video_right, video_torso"
    )


_camera_readers_lock = threading.Lock()
_camera_readers = {}
_legacy_right_alias_users = 0


def _publish_legacy_frame(name, frame, number, stamp):
    global captured_img_l, captured_img_h, captured_img_r, captured_img_t
    global captured_frame_number_l, captured_frame_number_h, captured_frame_number_r, captured_frame_number_t
    global captured_timestamp_l, captured_timestamp_h, captured_timestamp_r, captured_timestamp_t
    if name == 'left' or (name == 'right' and _legacy_right_alias_users):
        captured_img_l, captured_frame_number_l, captured_timestamp_l = frame, number, stamp
    if name == 'overhead':
        captured_img_h, captured_frame_number_h, captured_timestamp_h = frame, number, stamp
    elif name == 'right':
        captured_img_r, captured_frame_number_r, captured_timestamp_r = frame, number, stamp
    elif name == 'torso':
        captured_img_t, captured_frame_number_t, captured_timestamp_t = frame, number, stamp


@dataclass
class _CameraWorker:
    stop: threading.Event
    thread: object = None
    users: int = 0
    error: Exception | None = None


def _camera_reader_main(name, worker):
    reader = None
    try:
        reader = SharedFrameReader(name)
        while not worker.stop.is_set():
            frame, number, stamp = reader.read_frame()
            if frame is not None:
                _publish_legacy_frame(name, frame, number, stamp)
            worker.stop.wait(.001)
    except Exception as exc:
        worker.error = exc
    finally:
        if reader is not None:
            reader.close()


class _PythonCameraReaderLease:
    """One independent reader per camera per process, shared by all consumers.

    Slow/missing cameras cannot hold up another camera's reader. The lock covers
    worker ownership only; packet publication uses a separate, short lock.
    """
    def __init__(self, camera_names, *, legacy_right_alias=False):
        global _legacy_right_alias_users
        self.names = list(dict.fromkeys(name.removeprefix('video_') for name in camera_names))
        self.closed = False
        self.legacy_right_alias = legacy_right_alias
        self.workers = {}
        with _camera_readers_lock:
            if legacy_right_alias:
                _legacy_right_alias_users += 1
            for name in self.names:
                worker = _camera_readers.get(name)
                if worker is None:
                    with _camera_packet_lock:
                        captured_camera_packets.pop('video_' + name, None)
                    worker = _CameraWorker(threading.Event())
                    worker.thread = threading.Thread(target=_camera_reader_main,
                        args=(name, worker), name=f'camera-reader-{name}', daemon=True)
                    _camera_readers[name] = worker
                    worker.thread.start()
                worker.users += 1
                self.workers[name] = worker

    def check_health(self):
        for name, worker in self.workers.items():
            if worker.error is not None:
                raise RuntimeError(f'Camera reader {name} failed: {worker.error}') from worker.error

    def wait_ready(self, timeout=5.):
        deadline = time_module.monotonic() + timeout
        while True:
            self.check_health()
            try:
                get_camera_observation(['video_' + name for name in self.names])
                return
            except RuntimeError:
                if time_module.monotonic() >= deadline:
                    raise
            # Startup only. Per-observation snapshots never wait for a camera.
            time_module.sleep(.005)

    def close(self):
        global _legacy_right_alias_users
        if self.closed:
            return
        with _camera_readers_lock:
            self.closed = True
            if self.legacy_right_alias:
                _legacy_right_alias_users -= 1
            for name, worker in self.workers.items():
                worker.users -= 1
                if not worker.users:
                    worker.stop.set()
                    worker.thread.join(timeout=1.)
                    if worker.thread.is_alive():
                        # Keep ownership so another lease cannot start a competing reader.
                        continue
                    del _camera_readers[name]
                    with _camera_packet_lock:
                        captured_camera_packets.pop('video_' + name, None)


class CameraReaderLease:
    """Shared native cache by default; explicit Python backend for comparison."""
    def __new__(cls, camera_names, *, legacy_right_alias=False):
        backend = os.environ.get('FLASH_ALOHA_CAMERA_CACHE', 'robotmq')
        if backend == 'python':
            if robotmq_camera_cache.active_cache() is not None:
                raise RuntimeError('Cannot change camera cache backend while readers are active')
            return _PythonCameraReaderLease(camera_names, legacy_right_alias=legacy_right_alias)
        if backend != 'robotmq':
            raise ValueError('FLASH_ALOHA_CAMERA_CACHE must be robotmq or python')
        with _camera_readers_lock:
            if _camera_readers:
                raise RuntimeError('Cannot change camera cache backend while readers are active')
            specs = []
            for short in dict.fromkeys(name.removeprefix('video_') for name in camera_names):
                if not short or '/' in short or short in ('.', '..'):
                    raise ValueError(f'Invalid camera name: {short}')
                config = resolve_camera_config(name=short, device_path=f'/dev/video_{short}',
                    fallback_width=DEFAULT_WIDTH, fallback_height=DEFAULT_HEIGHT)
                specs.append(robotmq_camera_cache.CameraSpec('video_' + short,
                    f'/dev/shm/{SHM_PREFIX}{short}', config.width, config.height))
            return robotmq_camera_cache.CameraLease(specs)


def acquire_camera_readers(camera_names, *, timeout=5.):
    lease = CameraReaderLease(camera_names)
    try:
        lease.wait_ready(timeout)
    except BaseException:
        lease.close()
        raise
    return lease


def capture_image_from_shared_memory(stop_event, double_arm=False, camera_side='left', camera_names=None):
    """Compatibility entry point: supervise independently running camera readers."""
    if camera_names is not None:
        names = camera_names
    elif double_arm:
        names = ['left', 'overhead', 'right']
    elif camera_side in ('left', 'right'):
        names = [camera_side, 'overhead']
    else:
        raise ValueError(f'Unknown camera side: {camera_side}')
    lease = CameraReaderLease(names, legacy_right_alias=(
        camera_names is None and not double_arm and camera_side == 'right'))
    try:
        while not stop_event.wait(.05):
            lease.check_health()
            if isinstance(lease, robotmq_camera_cache.CameraLease):
                # Compatibility previews only. Standard observations read the
                # native cache directly, even while this Python thread is paused.
                try:
                    images, observation = lease.cache.snapshot(lease.names, copy_images=True)
                except CameraTimingError:
                    continue  # startup/stale preview; live capture fails closed
                for name, frame in images.items():
                    reference = observation['cameras'][name]
                    _publish_legacy_frame(name.removeprefix('video_'), frame,
                        reference['frame_number'], reference['capture_timestamp'])
                    if camera_names is None and not double_arm and camera_side == 'right' and name == 'video_right':
                        _publish_legacy_frame('left', frame, reference['frame_number'], reference['capture_timestamp'])
    finally:
        lease.close()

def print_camera_info(cameras_info):
    """
    Print camera information in a formatted way.
    
    Args:
        cameras_info: List of camera info dictionaries
    """
    fps_str = ", ".join([str(info['fps']) for info in cameras_info])
    autofocus_str = ", ".join([str(info.get('autofocus', 'N/A')) for info in cameras_info])
    width_str = ", ".join([str(info['width']) for info in cameras_info])
    height_str = ", ".join([str(info['height']) for info in cameras_info])
    
    print(f"fps:{fps_str},  autofocus:{autofocus_str}, width:{width_str}, height:{height_str}")

# Shared image file path for cross-process camera sharing
SHARED_OVERHEAD_IMAGE_PATH = '/dev/shm/flash_aloha_overhead.npy'

def capture_image(stop_event, double_arm=False, camera_side='left', camera_names=None):
    """
    Capture images from cameras in a loop.

    Args:
        stop_event: Threading event to signal when to stop
        double_arm: If True, capture from all three cameras (ignored if camera_names is provided)
        camera_side: For single arm mode, which side to use (ignored if camera_names is provided):
                     'left' - opens video_left + video_overhead
                     'right' - opens video_right + video_overhead
        camera_names: Optional list of full camera device names (e.g., ['video_left', 'video_overhead', 'video_torso']).
                      When provided, overrides double_arm/camera_side logic.

    Note: When camera_side='left', overhead images are saved to shared memory
          for other processes to read. When camera_side='right' and overhead
          camera fails to open, images are read from shared memory.
    """
    global captured_img_l
    global captured_img_h
    global captured_img_r
    global captured_img_t
    global captured_timestamp_l
    global captured_timestamp_h
    global captured_timestamp_r
    global captured_timestamp_t

    cameras = {}  # device_name -> cv2.VideoCapture
    cameras_info = []
    read_overhead_from_shared = False

    # Map full camera names to global variable names
    cam_name_to_device = {
        'video_left': '/dev/video_left',
        'video_overhead': '/dev/video_overhead',
        'video_right': '/dev/video_right',
        'video_torso': '/dev/video_torso',
    }

    if camera_names is not None:
        # camera_names mode: open exactly the specified cameras
        for cam_name in camera_names:
            device_path = cam_name_to_device.get(cam_name, f'/dev/{cam_name}')
            cap = init_camera(device_path)
            if cap:
                cameras[cam_name] = cap
                cameras_info.append(get_camera_info(cap, cam_name))
    elif double_arm:
        # Double arm mode: open all cameras
        for cam_name in ['video_left', 'video_overhead', 'video_right']:
            cap = init_camera(cam_name_to_device[cam_name])
            if cap:
                cameras[cam_name] = cap
                cameras_info.append(get_camera_info(cap, cam_name))
    elif camera_side == 'left':
        for cam_name in ['video_left', 'video_overhead']:
            cap = init_camera(cam_name_to_device[cam_name])
            if cap:
                cameras[cam_name] = cap
                cameras_info.append(get_camera_info(cap, cam_name))
    elif camera_side == 'right':
        cap_r = init_camera("/dev/video_right")
        if cap_r:
            cameras['video_right'] = cap_r
            cameras_info.append(get_camera_info(cap_r, "right"))
        cap_h = init_camera("/dev/video_overhead")
        if cap_h:
            cameras['video_overhead'] = cap_h
            cameras_info.append(get_camera_info(cap_h, "overhead"))
        else:
            print("[INFO] Overhead camera in use by another process - reading from shared memory")
            read_overhead_from_shared = True
    else:
        raise ValueError(f"Unknown camera_side: {camera_side}. Must be 'left' or 'right'.")

    if cameras_info:
        print_camera_info(cameras_info)
    else:
        print("[WARN] No cameras were successfully opened!")

    def _read_cam(cam_name):
        """Read a frame from a camera and store in the appropriate global."""
        global captured_img_l, captured_img_h, captured_img_r, captured_img_t
        global captured_timestamp_l, captured_timestamp_h, captured_timestamp_r, captured_timestamp_t
        cap = cameras.get(cam_name)
        if cap is None:
            return True
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame from {cam_name}.")
            return False
        timestamp = time_module.time()
        if cam_name == 'video_left':
            captured_img_l = frame
            captured_timestamp_l = timestamp
        elif cam_name == 'video_overhead':
            captured_img_h = frame
            captured_timestamp_h = timestamp
        elif cam_name == 'video_right':
            captured_img_r = frame
            captured_timestamp_r = timestamp
        elif cam_name == 'video_torso':
            captured_img_t = frame
            captured_timestamp_t = timestamp
        return True

    try:
        while not stop_event.is_set():
            if camera_names is not None:
                # camera_names mode: read all specified cameras
                for cam_name in camera_names:
                    if not _read_cam(cam_name):
                        return
            elif double_arm:
                for cam_name in ['video_left', 'video_right', 'video_overhead']:
                    if not _read_cam(cam_name):
                        return
            else:
                # Single arm legacy mode
                if camera_side == 'left':
                    if 'video_left' in cameras:
                        ret, captured_img_l = cameras['video_left'].read()
                        if not ret:
                            print("Error: Could not read frame from left camera.")
                            break
                        captured_timestamp_l = time_module.time()
                elif camera_side == 'right':
                    if 'video_right' in cameras:
                        ret, frame = cameras['video_right'].read()
                        if not ret:
                            print("Error: Could not read frame from right camera.")
                            break
                        captured_img_l = frame
                        captured_timestamp_l = time_module.time()

                # Handle overhead camera in legacy mode
                if 'video_overhead' in cameras:
                    ret, captured_img_h = cameras['video_overhead'].read()
                    if not ret:
                        print("Error: Could not read frame from overhead camera.")
                        break
                    captured_timestamp_h = time_module.time()
                    if camera_side == 'left' or double_arm:
                        try:
                            np.save(SHARED_OVERHEAD_IMAGE_PATH, captured_img_h)
                        except Exception:
                            pass
                elif read_overhead_from_shared:
                    try:
                        if os.path.exists(SHARED_OVERHEAD_IMAGE_PATH):
                            captured_img_h = np.load(SHARED_OVERHEAD_IMAGE_PATH)
                            captured_timestamp_h = time_module.time()    
                    except Exception:
                        pass
    finally:
        for cap in cameras.values():
            cap.release()

def get_camera_info(cam, cam_name):
    """
    Get camera information including fps, dimensions, and autofocus settings.
    
    Args:
        cam: cv2.VideoCapture object
        cam_name: Name of the camera for identification
    
    Returns:
        Dictionary containing camera properties
    """
    return {
        'name': cam_name,
        'fps': cam.get(cv2.CAP_PROP_FPS),
        'width': cam.get(cv2.CAP_PROP_FRAME_WIDTH),
        'height': cam.get(cv2.CAP_PROP_FRAME_HEIGHT),
        'autofocus': cam.get(cv2.CAP_PROP_AUTOFOCUS)
    }


def _get_camera_params_npz_path():
    env_path = os.environ.get("FLASH_ALOHA_CAMERA_PARAMS")
    if env_path:
        return env_path

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    calibration_path = os.path.join(repo_root, "calibration", "camera_params.npz")
    if os.path.exists(calibration_path):
        return calibration_path
    return os.path.join(repo_root, "camera_params.npz")


def _set_v4l2_control(device_path, name, value):
    import subprocess

    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device_path, f"--set-ctrl={name}={int(value)}"],
            capture_output=True,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


V4L2_CONTROL_ALIASES = {
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


OPENCV_CONTROL_PROPS = {
    "brightness": cv2.CAP_PROP_BRIGHTNESS,
    "contrast": cv2.CAP_PROP_CONTRAST,
    "saturation": cv2.CAP_PROP_SATURATION,
    "hue": cv2.CAP_PROP_HUE,
    "auto_wb": cv2.CAP_PROP_AUTO_WB,
    "gamma": cv2.CAP_PROP_GAMMA,
    "gain": cv2.CAP_PROP_GAIN,
    "wb_temperature": cv2.CAP_PROP_WB_TEMPERATURE,
    "sharpness": getattr(cv2, "CAP_PROP_SHARPNESS", -1),
    "backlight_compensation": getattr(cv2, "CAP_PROP_BACKLIGHT", -1),
    "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
    "exposure_absolute": cv2.CAP_PROP_EXPOSURE,
    "autofocus": cv2.CAP_PROP_AUTOFOCUS,
    "focus": cv2.CAP_PROP_FOCUS,
}


CAMERA_CONTROL_APPLY_ORDER = (
    "exposure_dynamic_framerate",
    "auto_exposure",
    "exposure_absolute",
    "auto_wb",
    "wb_temperature",
    "brightness",
    "contrast",
    "saturation",
    "hue",
    "gamma",
    "gain",
    "power_line_frequency",
    "sharpness",
    "backlight_compensation",
    "autofocus",
    "focus",
)


def _set_v4l2_control_by_key(device_path, key, value):
    for control_name in V4L2_CONTROL_ALIASES.get(key, (key,)):
        if _set_v4l2_control(device_path, control_name, value):
            return True
    return False


def _set_opencv_control(cap, key, value):
    prop = OPENCV_CONTROL_PROPS.get(key, -1)
    if prop is not None and prop >= 0:
        cap.set(prop, int(value))


def _apply_tuned_camera_settings(cap, device_path, settings):
    ordered_keys = list(CAMERA_CONTROL_APPLY_ORDER)
    ordered_keys.extend(key for key in settings if key not in ordered_keys)
    for key in ordered_keys:
        value = settings.get(key)
        if value is None:
            continue
        _set_v4l2_control_by_key(device_path, key, value)
        _set_opencv_control(cap, key, value)


def _load_tuned_camera_settings(device_path):
    params_path = _get_camera_params_npz_path()
    camera_name = os.path.basename(device_path).replace("video_", "")
    try:
        config = resolve_camera_config(camera_name, device_path, params_path=params_path)
    except Exception as exc:
        print(f"[WARN] Failed to load camera params from {params_path}: {exc}")
        return dict(DEFAULT_CONTROL_SETTINGS)
    return dict(config.settings)


def init_camera(device_path, width=1280, height=720, fps=60):
    """
    Initialize a camera with standard settings.
    
    Args:
        device_path: Path to the camera device (e.g., "/dev/video_left")
        width: Frame width (default: 1280)
        height: Frame height (default: 720)
        fps: Frames per second (default: 60)
    
    Returns:
        cv2.VideoCapture object or None if failed to open
    """
    if not os.path.exists(device_path):
        print(f"[WARN] {device_path} does not exist; skipping")
        return None

    cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERR] Failed to open {device_path}")
        return None

    tuned_settings = _load_tuned_camera_settings(device_path)

    # Stream format & timing
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    
    # if 'video_right' in device_path:
    #     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Global shutter camera max 720p
    # else:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Global Shutter Camera (video_right) supports 90fps and needs buffer=2 for full speed
    # Buffer=2 gives ~89 FPS with ~22ms max latency (best tradeoff)
    # Other cameras work best with buffer=1 at 60fps
    # if 'video_right' in device_path:
    #     # cap.set(cv2.CAP_PROP_FPS, 30)
    #     # global shutter
    #     cap.set(cv2.CAP_PROP_FPS, 90)
    #     # cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    # else:
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    _apply_tuned_camera_settings(cap, device_path, tuned_settings)

    return cap

