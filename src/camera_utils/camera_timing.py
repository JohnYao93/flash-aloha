"""Camera capture-clock contract shared by recording, conversion and inference.

Times are driver CLOCK_MONOTONIC seconds, never frame-count/FPS estimates.
The limits are configurable because a policy's tolerated observation delay is
part of its deployment contract, not a property of the video file format.
"""
from __future__ import annotations

# Support direct execution as well as package imports.
import sys as _path_sys
from pathlib import Path as _Path
_ACT_ROOT = next(p for p in _Path(__file__).resolve().parents if p.name == "src")
if str(_ACT_ROOT) not in _path_sys.path:
    _path_sys.path.insert(0, str(_ACT_ROOT))

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import struct
import time
import warnings

import numpy as np


class CameraTimingError(RuntimeError):
    pass


def publish_shared_clock_contract(shm, *, directory='/dev/shm'):
    """Bind the producer's clock declaration to this exact shared-memory inode."""
    stat = os.fstat(shm._fd)
    path = Path(directory) / f'{shm.name}.clock.json'
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(dict(schema_version=1, timestamp_source='v4l2_monotonic',
                                        device=stat.st_dev, inode=stat.st_ino, size=stat.st_size)))
    os.replace(temporary, path)


def validate_shared_clock_contract(shm, *, directory='/dev/shm'):
    """Inference must not accept an old synthetic clock merely because it is fresh."""
    path = Path(directory) / f'{shm.name}.clock.json'
    try:
        contract = json.loads(path.read_text())
        stat = os.fstat(shm._fd)
        valid = (contract.get('schema_version') == 1
                 and contract.get('timestamp_source') == 'v4l2_monotonic'
                 and (contract.get('device'), contract.get('inode'), contract.get('size'))
                 == (stat.st_dev, stat.st_ino, stat.st_size))
    except (OSError, ValueError):
        valid = False
    if not valid:
        raise CameraTimingError(f'{shm.name}: missing/stale capture-clock contract; restart camera_server.py')


@dataclass(frozen=True)
class TimingLimits:
    max_age_seconds: float = .050
    max_skew_seconds: float = .050
    max_future_seconds: float = .005

    def __post_init__(self):
        for value in asdict(self).values():
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Camera timing limits must be finite and positive')

    @classmethod
    def from_env(cls):
        return cls(*[float(os.environ.get(key, default)) / 1000 for key, default in (
            ('CAMERA_MAX_AGE_MS', '50'), ('CAMERA_MAX_SKEW_MS', '50'),
            ('CAMERA_MAX_FUTURE_MS', '5'))])


def check_camera_timestamps(timestamps, *, now=None, limits=None):
    """Check a contemporaneous bundle, including cameras equally far behind.

    This must not be called on asynchronous recorder heartbeat snapshots: their
    publication times differ. Use the actual inference bundle or shared headers.
    """
    limits = limits or TimingLimits.from_env()
    now = time.monotonic() if now is None else float(now)
    errors = []
    ages = {}
    for name, stamp in timestamps.items():
        if stamp is None or not math.isfinite(float(stamp)) or stamp <= 0:
            errors.append(f'{name}: missing/invalid driver capture timestamp')
            continue
        ages[name] = now - float(stamp)
        if ages[name] > limits.max_age_seconds:
            errors.append(f'{name}: frame age {ages[name]*1000:.1f} ms exceeds {limits.max_age_seconds*1000:g} ms')
        if ages[name] < -limits.max_future_seconds:
            errors.append(f'{name}: capture timestamp is {-ages[name]*1000:.1f} ms in the future (clock mismatch)')
    skew = max(ages.values()) - min(ages.values()) if ages else None
    if not timestamps:
        errors.append('No camera timestamps')
    if skew is not None and skew > limits.max_skew_seconds:
        errors.append(f'Camera capture-time spread {skew*1000:.1f} ms exceeds {limits.max_skew_seconds*1000:g} ms')
    return dict(alignable=not errors, checked_at=now, capture_timestamps=dict(timestamps),
                age_seconds=ages, skew_seconds=skew, errors=errors, limits=asdict(limits))


def require_camera_timestamps(timestamps, **kwargs):
    report = check_camera_timestamps(timestamps, **kwargs)
    if not report['alignable']:
        raise CameraTimingError('; '.join(report['errors']))
    return report


def driver_timestamp(timestamp, flags):
    """Reject unknown clocks and missing timestamps instead of inventing a time."""
    if not math.isfinite(timestamp) or timestamp <= 0:
        raise CameraTimingError('V4L2 returned no valid capture timestamp')
    if flags & 0xE000 != 0x2000:  # V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC
        raise CameraTimingError(f'V4L2 timestamp is not CLOCK_MONOTONIC (flags={flags:#x})')
    return float(timestamp)


def read_shared_camera_timestamps(camera_names):
    """Read only 16-byte headers; never copy images onto the robot writer path."""
    result = {}
    for name in camera_names:
        path = Path('/dev/shm') / ('flash_aloha_cam_' + name.removeprefix('video_'))
        try:
            with path.open('rb', buffering=0) as stream:
                stamp = None
                for _ in range(10):
                    stream.seek(0)
                    counter, candidate = struct.unpack('Qd', stream.read(16))
                    if not counter & (1 << 63) and math.isfinite(candidate):
                        stamp = candidate
                        break
                    time.sleep(.0001)
            result[name] = stamp
        except (OSError, struct.error):
            result[name] = None
    return result


def load_camera_timestamps(directory, name, group, *, timestamp_key=None):
    """Use driver timestamps from sidecars, including recordings made before v2."""
    path = Path(directory) / f'{name}.timing.jsonl'
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if [row['video_frame'] for row in rows] != list(range(len(rows))):
            raise CameraTimingError(f'{name}: non-contiguous video timing indices')
        stamps = np.asarray([row.get('capture_timestamp') for row in rows], dtype=np.float64)
        if len(stamps) != len(group['main_timestamps']):
            raise CameraTimingError(f'{name}: sidecar/video timestamp count mismatch')
    else:
        key = timestamp_key or ('main_timestamps' if 'main_timestamps' in group else 'timestamps')
        stamps = np.asarray(group[key], dtype=np.float64)
        if group.attrs.get('timestamp_source') != 'v4l2_monotonic':
            warnings.warn(f'{name}: legacy timestamps have unverified capture-clock provenance; '
                          'physical alignment cannot be certified', RuntimeWarning)
    if not len(stamps) or not np.isfinite(stamps).all() or np.any(np.diff(stamps) <= 0):
        raise CameraTimingError(f'{name}: capture timestamps must be finite and strictly increasing')
    return stamps


def selected_frame_indices(directory, name, control, *, sample_indices=None):
    """Resolve exact consumer references, or return None for a legacy recording.

    Never replace a missing/dropped referenced frame with a timestamp neighbor.
    Capture time also guards against frame-counter reuse after a camera restart.
    """
    if 'camera_observation' not in control:
        return None
    count = len(control['timestamps'])
    samples = np.arange(count) if sample_indices is None else np.asarray(sample_indices, dtype=int)
    try:
        group = control[f'camera_observation/{name}']
        for field in ('frame_number', 'capture_timestamp', 'consumer_received_timestamp'):
            if len(group[field]) != count:
                raise CameraTimingError(f'{name}: incomplete camera references')
        numbers = np.asarray(group['frame_number'])[samples]
        stamps = np.asarray(group['capture_timestamp'])[samples]
        rows = [json.loads(line) for line in (Path(directory) / f'{name}.timing.jsonl').read_text().splitlines()]
        lookup = {}
        for index, row in enumerate(rows):
            number = int(row['frame_number'])
            if number in lookup or row['video_frame'] != index:
                raise CameraTimingError(f'{name}: ambiguous frame IDs or invalid sidecar indices')
            lookup[number] = index
        indices = []
        continuity = np.asarray(control['continuity_id'])[samples]
        for number, stamp, segment in zip(numbers, stamps, continuity):
            index = lookup.get(int(number))
            if index is None:
                raise CameraTimingError(f'{name}: referenced frame {number} was not recorded (dropped or outside recording)')
            row = rows[index]
            if (not np.isfinite(stamp) or abs(float(row['capture_timestamp']) - stamp) > 1e-6
                    or int(row['continuity_id']) != segment):
                raise CameraTimingError(f'{name}: referenced frame {number} has mismatched capture time or continuity')
            indices.append(index)
        result = np.asarray(indices, dtype=np.int64)
        if np.any(np.diff(result) < 0):
            raise CameraTimingError(f'{name}: selected frame IDs move backwards')
        return result
    except KeyError as exc:
        raise CameraTimingError(f'{name}: missing camera observation metadata: {exc}') from exc
